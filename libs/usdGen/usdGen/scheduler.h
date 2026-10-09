// usdGen engine — scheduler: private tbb::task_arena + commit driver (03 §5.3/§5.4).
//
// I8: every parallel region runs in a PRIVATE task_arena sized at
// max(8, fast-core count) on heterogeneous Linux (the measured 8-thread
// knee of EV-001/EV-008 is the floor; all-fast-core width wins
// compute-bound passes while pinning holds), never the process-default
// 20-worker arena. USDGEN_THREAD_LIMIT (env) pins it explicitly;
// otherwise one-time calibration (03 §5.3) picks the smallest
// concurrency within 5 % of best.
// All loops run over the scheduler's worker pool — never pxr work::
// (WorkHasConcurrency() reads the process-global PXR_WORK_THREAD_LIMIT and would
// serialise under PXR_WORK_THREAD_LIMIT=1, 03 §5.3 caveat).
#ifndef USDGEN_SCHEDULER_H
#define USDGEN_SCHEDULER_H

#include "usdGen/graph.h"
#include "usdGen/op.h"
#include "usdGen/types.h"
#include "usdGen/workerPool.h"

#include <tbb/task_arena.h>
#include <tbb/task_scheduler_observer.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

/// Per-node timing/counters for one Run (03 §9.2; feeds UsdGenNodeStats).
struct UsdGenNodeRunStats
{
    UsdGenNodeId id = 0;
    double captureMs = 0.0;
    double evalMs = 0.0;
    uint64_t chunksEvaluated = 0;
};

struct UsdGenRunResult
{
    UsdGenCurveBuffer const *terminalOutput = nullptr;  // owned by the graph
    TfSpan<const UsdGenTileView> tiles;
    bool superseded = false;   // a newer request arrived; the published generation is unchanged
    bool topologyChanged = false; // re-partition happened; every tile re-interleaves
    UsdGenDiagnostics diagnostics; // node diagnostics for this run
    std::vector<UsdGenNodeRunStats> nodeStats;  // one entry per node with work
};

/// The work dispatcher handed to Capture() for intra-node parallelism (03 §8.1).
class UsdGenWorkDispatcher
{
public:
    explicit UsdGenWorkDispatcher(UsdGenWorkerPool *pool) : _pool(pool) {}
    /// Run a per-index body 0..count-1 over the scheduler's worker pool
    /// (capture phase). A null pool runs the body inline, serially.
    /// claimChunk overrides the pool's claim heuristic (0 keeps it); see
    /// UsdGenWorkerPool::ParallelFor.
    void ParallelFor(size_t count, void (*body)(size_t, void *), void *payload,
                     size_t claimChunk = 0);
    /// Pool worker count (1 when null/serial), so Capture() can size
    /// partitions instead of oversubscribing ranges onto one worker.
    int MaxConcurrency() const { return _pool ? _pool->Workers() : 1; }
private:
    UsdGenWorkerPool *_pool;
};

class UsdGenScheduler
{
public:
    using TileCompleted = std::function<void(UsdGenTileView const&,
                                             UsdGenCurveBuffer const&)>;
    /// threadLimit: 0 == env USDGEN_THREAD_LIMIT, else one-shot calibration.
    explicit UsdGenScheduler(int threadLimit = 0);
    ~UsdGenScheduler();

    UsdGenScheduler(const UsdGenScheduler &) = delete;
    UsdGenScheduler &operator=(const UsdGenScheduler &) = delete;

    int ThreadLimit() const noexcept;
    tbb::task_arena const &Arena() const noexcept { return _arena; }
    UsdGenWorkDispatcher MakeWorkDispatcher() const;

    /// The 8-step commit of 03 §5.4 (steps 1-6 engine-side; step 7 diff and
    /// step 8 notices are the session/imaging side). Evaluates every node in
    /// topological order with its dirty chunks, runs the reference lane first,
    /// interleaves dirty tiles, and checks for supersession between nodes.
    UsdGenRunResult Run(UsdGenGraph &graph, UsdGenEvalContext const &evalCtx,
                        uint64_t generationRequested,
                        TileCompleted tileCompleted = {});

private:
    // Retained for the Arena() accessor; parallel regions run on _pool.
    tbb::task_arena _arena;
    int _threadLimit;
    // Heterogeneous-core placement (tbbFastCores.h): on big.LITTLE Linux
    // the arena workers prefer max-frequency cores. Null when homogeneous,
    // undetectable, or disabled. Declared after _arena so it detaches first.
    std::unique_ptr<tbb::task_scheduler_observer> _affinityObserver;
    // Persistent fork/join pool behind MakeWorkDispatcher (workerPool.h):
    // pinned workers parked on a generation counter, without the arena's
    // per-region wakeup cost. Declared last so it joins before the arena.
    UsdGenWorkerPool _pool;
};

/// One-shot process calibration (03 §5.3): sweeps concurrencies {2,4,8,16,
/// min(20,physical)} over a fixed synthetic styler pass (64 chunks ≈ 2 MB)
/// and returns the smallest concurrency within 5 % of best, clamped [2,physical].
int CalibrateThreads();

}  // namespace usdGen

#endif  // USDGEN_SCHEDULER_H
