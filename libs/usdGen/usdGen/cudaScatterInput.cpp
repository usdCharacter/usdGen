#include "usdGen/cudaScatterInput.h"

#include "usdGen/opRegistry.h"

#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <vector>

namespace usdGen {
namespace {

CudaScatterInputStatus Fail(CudaScatterInputStatus status, std::string message,
                            std::string* reason) {
    if (reason) *reason = std::move(message);
    return status;
}

bool Finite(GfVec3f const& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}
bool Finite(GfVec2f const& v) { return std::isfinite(v[0]) && std::isfinite(v[1]); }

// Scatter is a generator: it declares only density and flip.
bool AllowedScatterParam(TfToken const& name) {
    return name == TfToken("density") || name == TfToken("flip") || name == TfToken("subdivisionLevel");
}

bool IsFloat(VtValue const& v) { return v.IsHolding<float>() || v.IsHolding<double>(); }

bool ValidScatterParam(UsdGenParamValue const& param) {
    if(param.name==TfToken("subdivisionLevel"))
        return param.value.IsHolding<int>() && param.value.UncheckedGet<int>()==0;
    if (param.name == TfToken("density")) return IsFloat(param.value);
    return param.name == TfToken("flip") && param.value.IsHolding<bool>();
}

// Threaded validation scans (verdict-identical): every ValidateSurface
// group below reports the same message for any failing element, and the
// groups keep their serial order with the same early exit between groups,
// so chunking within a group only changes how fast the verdict arrives.
// Small arrays stay serial (dispatch costs more than the scan below ~32K
// elements). Plain TBB: validation runs outside any scheduler arena.
template <class T, class Bad>
bool ValidateAnyBad(T const* data, size_t n, Bad bad) {
    int const workers = tbb::this_task_arena::max_concurrency();
    size_t const chunks =
        (workers > 1 && n > 32768) ? std::min({size_t(workers), size_t(8), n})
                                   : 1;
    if (chunks == 1) {
        for (size_t i = 0; i < n; ++i)
            if (bad(data[i])) return true;
        return false;
    }
    std::vector<unsigned char> partial(chunks, 0);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, chunks),
        [&](tbb::blocked_range<size_t> const& range) {
            for (size_t c = range.begin(); c != range.end(); ++c) {
                size_t const i0 = (c * n) / chunks;
                size_t const i1 = ((c + 1) * n) / chunks;
                bool hit = false;
                for (size_t i = i0; i < i1; ++i) {
                    if (bad(data[i])) { hit = true; break; }
                }
                partial[c] = hit ? 1 : 0;
            }
        });
    for (unsigned char f : partial)
        if (f) return true;
    return false;
}

// Face-count validation with the exact corner total (verdict-identical):
// each chunk scans its slice with the serial spelling (count<3 fails,
// overflow-checked accumulation), then the chunk sums combine in chunk
// order with the same overflow check. All terms are non-negative past
// the count<3 filter, so the running total overflows exactly when the
// grand total exceeds the maximum, and any count<3 fails either way.
// Same verdict, same total, any chunking.
bool ValidateCounts(int const* counts, size_t n, size_t* corners) {
    constexpr size_t kMax = std::numeric_limits<size_t>::max();
    int const workers = tbb::this_task_arena::max_concurrency();
    size_t const chunks =
        (workers > 1 && n > 32768) ? std::min({size_t(workers), size_t(8), n})
                                   : 1;
    if (chunks == 1) {
        size_t total = 0;
        for (size_t i = 0; i < n; ++i) {
            int const count = counts[i];
            if (count < 3 || size_t(count) > kMax - total) return false;
            total += size_t(count);
        }
        *corners = total;
        return true;
    }
    struct CountChunk { bool bad = false; size_t sum = 0; };
    std::vector<CountChunk> partial(chunks);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, chunks),
        [&](tbb::blocked_range<size_t> const& range) {
            for (size_t c = range.begin(); c != range.end(); ++c) {
                size_t const i0 = (c * n) / chunks;
                size_t const i1 = ((c + 1) * n) / chunks;
                size_t acc = 0;
                bool bad = false;
                for (size_t i = i0; i < i1; ++i) {
                    int const count = counts[i];
                    if (count < 3 || size_t(count) > kMax - acc) {
                        bad = true;
                        break;
                    }
                    acc += size_t(count);
                }
                partial[c].bad = bad;
                partial[c].sum = acc;
            }
        });
    size_t total = 0;
    for (auto const& p : partial) {
        if (p.bad) return false;
        if (p.sum > kMax - total) return false;
        total += p.sum;
    }
    *corners = total;
    return true;
}

CudaScatterInputStatus ValidateSurface(UsdGenSurfaceDesc const& surface,
                                       std::string* reason) {
    if (surface.faceVertexCounts.empty()) {
        if (!surface.faceVertexIndices.empty())
            return Fail(CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has indices but no faces", reason);
        if (!surface.uv.empty() && surface.uv.size() != surface.restPoints.size())
            return Fail(CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface UV cardinality differs from rest positions", reason);
        if (ValidateAnyBad(surface.restPoints.cdata(), surface.restPoints.size(),
                           [](GfVec3f const& p) { return !Finite(p); }))
            return Fail(CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has non-finite rest position", reason);
        if (ValidateAnyBad(surface.uv.cdata(), surface.uv.size(),
                           [](GfVec2f const& uv) { return !Finite(uv); }))
            return Fail(CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has non-finite UV", reason);
        return CudaScatterInputStatus::Ok;
    }
    if (surface.restPoints.empty())
        return Fail(CudaScatterInputStatus::InvalidSurface,
                    "Scatter surface faces require rest positions", reason);
    if (!surface.uv.empty() && surface.uv.size() != surface.restPoints.size())
        return Fail(CudaScatterInputStatus::InvalidSurface,
                    "Scatter surface UV cardinality differs from rest positions", reason);
    size_t corners = 0;
    if (!ValidateCounts(surface.faceVertexCounts.cdata(),
                        surface.faceVertexCounts.size(), &corners))
        return Fail(CudaScatterInputStatus::InvalidSurface,
                    "Scatter surface has invalid face cardinality", reason);
    if (corners != surface.faceVertexIndices.size())
        return Fail(CudaScatterInputStatus::InvalidSurface,
                    "Scatter surface face-index cardinality differs from face counts", reason);
    if (ValidateAnyBad(surface.restPoints.cdata(), surface.restPoints.size(),
                       [](GfVec3f const& p) { return !Finite(p); }))
        return Fail(CudaScatterInputStatus::InvalidSurface,
                    "Scatter surface has non-finite rest position", reason);
    if (ValidateAnyBad(surface.uv.cdata(), surface.uv.size(),
                       [](GfVec2f const& uv) { return !Finite(uv); }))
        return Fail(CudaScatterInputStatus::InvalidSurface,
                    "Scatter surface has non-finite UV", reason);
    size_t const points = surface.restPoints.size();
    if (ValidateAnyBad(surface.faceVertexIndices.cdata(),
                       surface.faceVertexIndices.size(),
                       [points](int index) {
                           return index < 0 || size_t(index) >= points;
                       }))
        return Fail(CudaScatterInputStatus::InvalidSurface,
                    "Scatter surface has out-of-range face index", reason);
    // The subset loop stays serial: range-vs-duplicate precedence is
    // element-order-dependent (the first failing element names the
    // message), and subsets are small (usually empty).
    std::set<int> subset;
    for (int face : surface.subsetFaces)
        if (face < 0 || size_t(face) >= surface.faceVertexCounts.size())
            return Fail(CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has out-of-range subset face", reason);
        else if (!subset.insert(face).second)
            return Fail(CudaScatterInputStatus::InvalidSurface,
                        "Scatter surface has duplicate subset face", reason);
    return CudaScatterInputStatus::Ok;
}

} // namespace

namespace {
// Pooled convert shells (bit-identical): the convert's positions plane
// churns ~12MB of alloc+fault per 1M-root convert, and the transpose
// fill fully overwrites it, so a released shell's positions capacity
// is purely a warm backing store with no observable contents. (The six
// adopted planes move out of the capture buffer, so they never touch
// shell storage.) Checkout is one shell per thread (depth 1: sequential
// converts, the bench and the session task path, always hit; deeper
// pipelines allocate fresh, exactly as before). The shared_ptr's deleter
// recycles into the RELEASING thread's pool, so cross-thread handoff
// (convert here, release after grow commit elsewhere) migrates shells
// instead of racing. Bounded: one idle shell per converting thread;
// overflow frees as unpooled. Deliberately depth-1: the ~270MB emission
// pool measurably slowed downstream fresh allocs and was reverted
// (scatter.cpp); this retains at most one convert per thread.
thread_local std::unique_ptr<gpu::ScatterGrowRoots> t_convertShell;
// Pooled capture shells (bit-identical): PrepareCudaScatterInput builds a
// fresh capture per call and destroys it after the convert, churning the
// touched VtArray pages through the kernel's page-table teardown every
// call (~100% sys time by rusage split; the process runs with a raised
// dynamic mmap threshold, so the churn spells as brk growth/contraction
// rather than munmap). Capture already clear()s every plane it fills
// before writing (the gather leg's resize is exact), and it never reads
// incoming plane contents, so a cleared shell is indistinguishable from
// fresh storage. The six adopted planes move out to the roots and come
// back by donation when the roots release (see RecycleConvertShell), so
// a sequential stream never mallocs them twice; only an overlapped
// pipeline (a live roots while the next capture checks out) allocates
// fresh, malloc-only through the uninitialized-fill resize. Depth-1
// per thread, same rationale as the convert shell; checkout and recycle
// both happen inside PrepareCudaScatterInput on the same thread (the
// capture never escapes), so no deleter handoff is needed. Failure
// paths destroy instead of recycling.
thread_local std::unique_ptr<UsdGenCapture> t_captureShell;
void RecycleConvertShell(gpu::ScatterGrowRoots const* roots)
{
    std::unique_ptr<gpu::ScatterGrowRoots> shell(
        const_cast<gpu::ScatterGrowRoots*>(roots));
    // Donate the adopted planes back to the idle capture shell (O(1)
    // moves): the convert adopted them out of a capture, so returning
    // them keeps the next checkout fully warm instead of mallocing six
    // fresh planes per call. t_captureShell is null exactly while a
    // capture is checked out on this thread (including the *out-overwrite
    // release mid-Prepare), so a present shell is always idle and safe
    // to refill; a missing one frees the planes as unpooled. COW-shared
    // planes (only tests copy roots) stay correct: the next Capture
    // resize detaches before writing, preserving the live copy's bytes.
    if (t_captureShell) {
        UsdGenCurveBuffer& buf = t_captureShell->MutableBuffer();
        shell->stableIds.donate(buf.curveId);
        shell->rootPrim.donate(buf.rootPrim);
        shell->rootUV.donate(buf.rootUV);
        shell->rootT.donate(buf.rootT);
        shell->rootB.donate(buf.rootB);
        shell->rootN.donate(buf.rootN);
        // The donated planes arrive sized; the shell must read empty
        // (same invariant the recycle block establishes). clear() keeps
        // the warm buffers when uniquely held, detaches when COW-shared.
        buf.curveId.clear();
        buf.rootPrim.clear();
        buf.rootUV.clear();
        buf.rootT.clear();
        buf.rootB.clear();
        buf.rootN.clear();
    }
    // Reset sizes, keep capacity: the n==0 convert skips every guarded
    // assign, so a recycled shell must read empty, not stale. (An
    // un-cleared pool returned 1M stale positions alongside 0 stableIds
    // for empty converts, breaking session estimates downstream.)
    shell->positions.clear();
    shell->stableIds.clear();
    shell->rootPrim.clear();
    shell->rootUV.clear();
    shell->rootT.clear();
    shell->rootB.clear();
    shell->rootN.clear();
    if (!t_convertShell)
        t_convertShell = std::move(shell);
    // else pool occupied: shell frees here, as unpooled.
}
} // namespace

CudaScatterInputStatus PrepareCudaScatterInput(
    UsdGenGraphDesc const& desc, SdfPath const& scatterPath,
    std::shared_ptr<const gpu::ScatterGrowRoots>* out, std::string* reason) {
    if (reason) reason->clear();
    if (!out || scatterPath.IsEmpty())
        return Fail(CudaScatterInputStatus::InvalidArgument,
                    "Scatter input needs an output owner and node path", reason);
    auto nodeIt = std::find_if(desc.nodes.begin(), desc.nodes.end(),
        [&](UsdGenNodeDesc const& node) { return node.path == scatterPath; });
    if (nodeIt == desc.nodes.end() || nodeIt->type != TfToken("UsdGenScatter"))
        return Fail(CudaScatterInputStatus::InvalidArgument,
                    "Scatter input path does not resolve to a Scatter node", reason);
    UsdGenNodeDesc const& node = *nodeIt;
    // 02 §6.3: a disabled generator publishes an EMPTY curve set, so the
    // description publishes no curves. Every operator downstream of the empty
    // root set is a no-op, which is exactly the CPU lane's behaviour. Any
    // disabled generator in the description empties this root set: Grow and
    // CurveSource are fused into the same native slice and have no separate
    // producer to empty.
    {
        usdGenRegisterM1Operators();
        bool generatorDisabled = false;
        for (UsdGenNodeDesc const& candidate : desc.nodes) {
            if (candidate.enabled) continue;
            std::unique_ptr<UsdGenOp> probe =
                UsdGenOpRegistry::Get().Create(candidate.type);
            generatorDisabled = generatorDisabled || (probe && probe->IsGenerator());
        }
        if (generatorDisabled) {
            *out = std::make_shared<gpu::ScatterGrowRoots>();
            return CudaScatterInputStatus::Ok;
        }
    }
    if (!node.mode.IsEmpty())
        return Fail(CudaScatterInputStatus::Unsupported,
                    "Scatter has no usdGen:mode property; it is always random", reason);
    if (!node.inputs.empty() || !node.references.empty() || !node.curves.empty() ||
        !node.maps.empty() || !node.mapBindings.empty() || !node.expressionBindings.empty() ||
        !node.ramps.empty())
        return Fail(CudaScatterInputStatus::Unsupported,
                    "Scatter input does not support geometry/reference/map/expression/ramp controls", reason);
    if (node.surfaces.size() != 1)
        return Fail(CudaScatterInputStatus::InvalidArgument,
                    "Scatter input requires exactly one surface path", reason);
    std::set<TfToken> seen;
    for (UsdGenParamValue const& param : node.params) {
        if (param.animated || !seen.insert(param.name).second ||
            !AllowedScatterParam(param.name) || !ValidScatterParam(param))
            return Fail(CudaScatterInputStatus::Unsupported,
                        "Scatter input has unsupported, duplicate, or animated parameter '" +
                        param.name.GetString() + "'", reason);
    }
    auto surfaceIt = std::find_if(desc.surfaces.begin(), desc.surfaces.end(),
        [&](UsdGenSurfaceDesc const& surface) { return surface.path == node.surfaces.front(); });
    if (surfaceIt == desc.surfaces.end())
        return Fail(CudaScatterInputStatus::InvalidSurface,
                    "Scatter input cannot resolve its surface path", reason);
    CudaScatterInputStatus validated = ValidateSurface(*surfaceIt, reason);
    if (validated != CudaScatterInputStatus::Ok) return validated;
    if (surfaceIt->restFromCurrentPoints)
        return Fail(CudaScatterInputStatus::Unsupported,
                    "Scatter input requires authored rest points, not current points", reason);

    UsdGenParamView params{&desc, &node};
    if (params.GetBool(TfToken("flip"), false))
        return Fail(CudaScatterInputStatus::Unsupported,
                    "Scatter input does not yet support flip frame handedness", reason);
    std::unique_ptr<UsdGenOp> op = UsdGenOpRegistry::Get().Create(TfToken("UsdGenScatter"));
    if (!op || op->GeometryInputArity() != 0)
        return Fail(CudaScatterInputStatus::Unsupported,
                    "Scatter input cannot resolve the authoritative random Scatter kernel", reason);
    UsdGenDiagnostics diagnostics;
    if (!op->Bind(params, &diagnostics))
        return Fail(CudaScatterInputStatus::CaptureFailed,
                    diagnostics.errors.empty() ? "Scatter Bind failed" : diagnostics.errors.front(), reason);
    // Pooled shell (see above): checkout the thread's idle capture or
    // create fresh. Same null failure, same message, either way.
    std::unique_ptr<UsdGenCapture> capture = std::move(t_captureShell);
    if (!capture)
        capture = op->CreateCapture();
    if (!capture)
        return Fail(CudaScatterInputStatus::CaptureFailed,
                    "Scatter kernel did not create capture storage", reason);
    UsdGenCurveBuffer emptyUpstream;
    UsdGenCaptureContext context;
    context.desc = &desc;
    context.params = &params;
    context.surface = static_cast<UsdGenSurfaceId>(surfaceIt - desc.surfaces.begin());
    context.seed = static_cast<uint32_t>(node.seed);
    context.diag = &diagnostics;
    if (!op->Capture(context, emptyUpstream, capture.get(), &diagnostics))
        return Fail(CudaScatterInputStatus::CaptureFailed,
                    diagnostics.errors.empty() ? "Scatter Capture failed" : diagnostics.errors.front(), reason);
    UsdGenCurveBuffer const& roots = capture->Buffer();
    size_t const n = roots.totalCurves;
    if (roots.totalCvs != n || roots.px.size() != n || roots.py.size() != n ||
        roots.pz.size() != n || roots.curveId.size() != n || roots.rootPrim.size() != n ||
        roots.rootUV.size() != n || roots.rootT.size() != n || roots.rootB.size() != n ||
        roots.rootN.size() != n)
        return Fail(CudaScatterInputStatus::CaptureFailed,
                    "Scatter Capture produced an invalid root topology", reason);
    // Pooled shell (see above): checkout the thread's idle shell or
    // allocate fresh. Held uniquely until *out publishes, so a throwing
    // assign() frees it exactly like the old make_shared path.
    std::unique_ptr<gpu::ScatterGrowRoots> shell;
    if (t_convertShell)
        shell = std::move(t_convertShell);
    else
        shell = std::make_unique<gpu::ScatterGrowRoots>();
    auto* prep = shell.get();
    // Adopted planes (bit-identical): the six layout-identical planes move
    // out of the capture buffer in O(1) instead of copying element-wise.
    // VtArray moves steal the buffer pointer, so the roots own the exact
    // bytes the capture gathered; the deleter donates them back to the
    // idle capture shell (see RecycleConvertShell), so sequential
    // converts reuse every buffer. Only positions still copies: it
    // transposes the SoA point planes the moves leave behind, through an
    // uninitialized fill over reused-or-fresh storage (a vector's resize
    // would value-init first). Every output slot is an independent
    // function of its three input floats into a disjoint lane, so the
    // fill chunks over workers for big inputs (plain TBB: this runs
    // outside any scheduler arena) and serially below the threshold.
    // Same bytes, any chunking.
    {
        UsdGenCurveBuffer& mut = capture->MutableBuffer();
        prep->stableIds.adopt(std::move(mut.curveId));
        prep->rootPrim.adopt(std::move(mut.rootPrim));
        prep->rootUV.adopt(std::move(mut.rootUV));
        prep->rootT.adopt(std::move(mut.rootT));
        prep->rootB.adopt(std::move(mut.rootB));
        prep->rootN.adopt(std::move(mut.rootN));
    }
    if (n) {
        float const* px = roots.px.cdata();
        float const* py = roots.py.cdata();
        float const* pz = roots.pz.cdata();
        int const transposeWorkers = tbb::this_task_arena::max_concurrency();
        size_t const transposeChunks =
            (transposeWorkers > 1 && n > 32768)
                ? std::min({size_t(transposeWorkers), size_t(8), n})
                : 1;
        prep->positions.resize_fill(
            n, [px, py, pz, transposeChunks](GfVec3f* b, GfVec3f* e) {
                if (transposeChunks == 1) {
                    size_t i = 0;
                    for (GfVec3f* d = b; d != e; ++d, ++i)
                        new (d) GfVec3f(px[i], py[i], pz[i]);
                    return;
                }
                size_t const M = size_t(e - b);
                tbb::parallel_for(
                    tbb::blocked_range<size_t>(0, transposeChunks),
                    [&](tbb::blocked_range<size_t> const& range) {
                        for (size_t c = range.begin(); c != range.end();
                             ++c) {
                            size_t const i0 = (c * M) / transposeChunks;
                            size_t const i1 = ((c + 1) * M) / transposeChunks;
                            for (size_t i = i0; i < i1; ++i)
                                new (b + i) GfVec3f(px[i], py[i], pz[i]);
                        }
                    });
            });
    }
    *out = std::shared_ptr<gpu::ScatterGrowRoots const>(
        shell.release(), RecycleConvertShell);
    // Recycle the capture shell (see above): every plane is clear()ed
    // (capacity kept), so the next checkout reads exactly like a fresh
    // capture, including the empty-topology early return (stale sizes
    // would trip the convert's topology validation).
    {
        UsdGenCurveBuffer& buf = capture->MutableBuffer();
        buf.px.clear(); buf.py.clear(); buf.pz.clear();
        buf.rest.clear(); buf.width.clear(); buf.hairT.clear();
        buf.extraCv.clear();
        buf.curveId.clear();
        buf.rootPrim.clear(); buf.rootUV.clear();
        buf.rootT.clear(); buf.rootN.clear(); buf.rootB.clear();
        buf.cvOffsets.clear(); buf.extraCurve.clear();
        buf.chunks.clear();
        buf.totalCurves = 0; buf.totalCvs = 0;
        buf.topologyVersion = 0; buf.valueVersion = 0;
    }
    t_captureShell = std::move(capture);
    return CudaScatterInputStatus::Ok;
}

} // namespace usdGen
