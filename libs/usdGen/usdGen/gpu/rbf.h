#ifndef USDGEN_GPU_RBF_H
#define USDGEN_GPU_RBF_H

#include "deviceBuffer.h"
#include <cusolverDn.h>
#include <cstdint>
#include <memory>
#include <string>

namespace usdGen { namespace gpu {
enum class RbfStatus { Ok, InvalidArgument, NonFiniteInput, RankDeficient, CudaError, SolverError };
// Test-only fresh-path fault seams.  Each affects one matching Begin call.
void TestFailNextFreshRbfBindAllocation() noexcept;
void TestFailNextFreshRbfBindAfterSubmit() noexcept;
void TestFailNextFreshRbfSolveAllocation() noexcept;
void TestFailNextFreshRbfSolveAfterInputSubmit() noexcept;
void TestFailNextFreshRbfSolveAfterSolverSubmit() noexcept;
void TestFailNextFreshRbfEvaluateAllocation() noexcept;
void TestFailNextFreshRbfEvaluateAfterSubmit() noexcept;
void TestFailNextFreshRbfResolvePreflight() noexcept;
void TestFailNextFreshRbfResolveCommit() noexcept;
uint64_t FreshRbfAcceptAttemptCountForTesting() noexcept;
uint64_t FreshRbfRollbackAttemptCountForTesting() noexcept;
// Test-only direct-evaluate cache seam. While disabled, Evaluate runs the
// uncached kernel and leaves the cache state untouched, so a test can
// compare the cached and uncached paths bitwise on the same binding.
void TestDisableCudaRbfEvalCache(bool disable) noexcept;
// Test-only direct-evaluate cache path counters (global across bindings;
// tests assert deltas). A hit evaluates through the cache, a miss refills
// it first; the over-cap fallback increments neither.
uint64_t CudaRbfEvalCacheHitsForTesting() noexcept;
uint64_t CudaRbfEvalCacheMissesForTesting() noexcept;
// Test-only direct-bind memo seam. While disabled, Bind always runs the
// factor slice, so a test can compare the memo-hit and full-slice paths
// bitwise on the same binding. The counters (global across bindings;
// tests assert deltas) count digest-probe hits (slice skipped) and
// digest-probe misses (slice ran); a bind that skips the probe (first
// bind, or a shape/smoothing key mismatch) increments neither.
void TestDisableCudaRbfBindMemo(bool disable) noexcept;
uint64_t CudaRbfBindMemoHitsForTesting() noexcept;
uint64_t CudaRbfBindMemoMissesForTesting() noexcept;

// Queries the selected CUDA implementation's legacy dense-LU workspace
// without allocating matrix storage or submitting device work.
cudaError_t GetCudaRbfLuWorkspaceElements(size_t sampleCount,
                                          size_t* elements) noexcept;

// GPU-only cubic RBF binding.  Bind and Solve copy only scalar diagnostics to
// the host; Evaluate has no geometry readback and is asynchronous.
class CudaRbfBinding {
public:
    CudaRbfBinding();
    ~CudaRbfBinding();
    CudaRbfBinding(const CudaRbfBinding&) = delete;
    CudaRbfBinding& operator=(const CudaRbfBinding&) = delete;
    RbfStatus Bind(DeviceView<const float3> restSamples, double smoothing, cudaStream_t stream);
    RbfStatus Solve(DeviceView<const float3> currentSamples, cudaStream_t stream);
    RbfStatus Evaluate(DeviceView<const float3> restCvs, DeviceView<float3> output, cudaStream_t stream);
    // Completes all Evaluate calls issued since the prior Finish.  Call before
    // publishing their output generation; only a scalar device flag is read.
    RbfStatus Finish(cudaStream_t stream);
    // Fresh rest binding is a private candidate.  Each Commit must be called
    // only after the owner has proved native terminal success and launcher
    // return. Commits are host-only; these low-level fresh primitives do not
    // alter the synchronous compatibility fields above.
    RbfStatus BeginFreshBind(DeviceView<const float3> restSamples, double smoothing, cudaStream_t stream,
                             UsdGenExecutionMemoryReservation* reservation = nullptr);
    RbfStatus CommitFreshBindExtent();
    RbfStatus BeginFreshBindRank(cudaStream_t stream);
    RbfStatus CommitFreshBindRank();
    RbfStatus BeginFreshBindLu(cudaStream_t stream);
    RbfStatus CommitFreshBindLu();
    // A fresh pose never writes the accepted coefficient buffer.  Its input
    // proof must succeed before the triangular solve can be submitted, and
    // the solver proof must succeed before CommitFreshSolve exchanges the
    // candidate coefficients into the accepted binding.
    RbfStatus BeginFreshSolve(DeviceView<const float3> currentSamples, cudaStream_t stream,
                              UsdGenExecutionMemoryReservation* reservation = nullptr);
    RbfStatus CommitFreshSolveInput();
    RbfStatus BeginFreshSolveFactors(cudaStream_t stream);
    RbfStatus CommitFreshSolve();
    // CommitFreshSolve makes the candidate coefficients visible solely to the
    // following fresh evaluation.  The parent must resolve that transaction
    // after all downstream work has terminal proof: Accept retains the new
    // pose, Rollback restores the prior accepted coefficients.  Neither call
    // submits CUDA work, waits, or frees a live CUDA allocation.
    RbfStatus AcceptFreshSolve();
    RbfStatus RollbackFreshSolve();
    // Pure transaction preflight.  A parent that owns both the surface pose
    // and solve candidate must check both sides before mutating either.
    bool CanAcceptFreshSolve() const noexcept;
    bool CanRollbackFreshSolve() const noexcept;
    RbfStatus BeginFreshEvaluate(DeviceView<const float3> restCvs, DeviceView<float3> output, cudaStream_t stream,
                                 UsdGenExecutionMemoryReservation* reservation = nullptr);
    RbfStatus CommitFreshEvaluate();
    void AbandonFresh() noexcept;
    bool HasUnprovenWork() const noexcept;
    // Candidate-only count; becomes useful to the following fresh solve
    // slice, without claiming that this bind has been published.
    size_t freshSampleCount() const;
    size_t sampleCount() const { return sampleCount_; }
    const char* diagnostic() const { return diagnostic_.c_str(); }
private:
    struct FreshState;
    struct DirectProofs;
    RbfStatus fail(RbfStatus status, const char* what);
    bool ensureProofs();
    // Direct-bind CUDA-graph replay: the bind slice (rest D2D, extent,
    // gram zero, gram, gram land, matrix build, LU) replays from one
    // instantiated graph while the capture key matches, collapsing one
    // launch gap per node into a single graph launch. Capture embeds
    // addresses and value arguments, so the key covers the shape, the
    // smoothing, the solver, and every address the slice reads or writes;
    // any mismatch re-captures, and any capture/launch failure runs the
    // same submits uncaptured (a capture only records, so the fallback
    // submits for real). The graph changes the launch vehicle only, so
    // every success-path byte is unchanged.
    struct BindGraphKey {
        int n = 0, m = 0;
        double smoothing = 0.0;
        cusolverDnHandle_t solver = nullptr;
        const void* src = nullptr;
        const void* rest = nullptr;
        const void* proof = nullptr;
        const void* params = nullptr;
        const void* gram = nullptr;
        const void* matrix = nullptr;
        const void* norm = nullptr;
        const void* work = nullptr;
        const void* pivots = nullptr;
        bool operator==(BindGraphKey const& o) const {
            return n == o.n && m == o.m && smoothing == o.smoothing &&
                solver == o.solver && src == o.src && rest == o.rest &&
                proof == o.proof && params == o.params && gram == o.gram &&
                matrix == o.matrix && norm == o.norm && work == o.work &&
                pivots == o.pivots;
        }
        bool operator!=(BindGraphKey const& o) const { return !(*this == o); }
    };
    struct BindGraph {
        cudaGraphExec_t exec = nullptr;
        BindGraphKey key;
        int consecutiveFailures = 0;
        bool disabled = false;
    };
    BindGraph bindGraph_;
    RbfStatus submitBindSlice(cudaStream_t stream, DeviceView<const float3> samples,
                              int n, int m, double smoothing);
    RbfStatus submitBindSliceGraphed(cudaStream_t stream, DeviceView<const float3> samples,
                                     int n, int m, double smoothing);
    // Direct-solve replay, same discipline as the bind graph: the solve
    // slice (posed D2D, RHS build, triangular solve) replays from one
    // instantiated graph while its key (shape, the 1/scale value argument,
    // solver, every address) matches; mismatch re-captures, failure runs
    // the same submits directly.
    struct SolveGraphKey {
        int n = 0, m = 0;
        double invScale = 0.0;
        cusolverDnHandle_t solver = nullptr;
        const void* src = nullptr;
        const void* current = nullptr;
        const void* rest = nullptr;
        const void* coefficients = nullptr;
        const void* proof = nullptr;
        const void* matrix = nullptr;
        const void* pivots = nullptr;
        bool operator==(SolveGraphKey const& o) const {
            return n == o.n && m == o.m && invScale == o.invScale &&
                solver == o.solver && src == o.src && current == o.current &&
                rest == o.rest && coefficients == o.coefficients &&
                proof == o.proof && matrix == o.matrix && pivots == o.pivots;
        }
        bool operator!=(SolveGraphKey const& o) const { return !(*this == o); }
    };
    struct SolveGraph {
        cudaGraphExec_t exec = nullptr;
        SolveGraphKey key;
        int consecutiveFailures = 0;
        bool disabled = false;
    };
    SolveGraph solveGraph_;
    RbfStatus submitSolveSlice(cudaStream_t stream, DeviceView<const float3> posed,
                               int n, int m, double invScale);
    RbfStatus submitSolveSliceGraphed(cudaStream_t stream, DeviceView<const float3> posed,
                                      int n, int m, double invScale);
    // Persisting-L2 window for the LU factors: cuSOLVER's triangular
    // solve reads the factors row-dependently, so L2-cold factors (the
    // prior evaluation's streaming evicts them) stall the dependent
    // chain. Pinning them persisting keeps them resident across
    // evaluations. Best-effort residency only: every failure runs
    // through unmodified with identical bytes, and re-arms only when
    // the stream or the factors move, so the steady state pays nothing.
    // One slot shared by the direct and fresh paths; alternating paths
    // re-arm (cold/test-only traffic, never the steady state).
    void armFactorL2Window(cudaStream_t stream, void const* factors, size_t factorBytes);
    cudaStream_t pl2Stream_ = nullptr;
    void const* pl2Ptr_ = nullptr;
    size_t pl2Bytes_ = 0;
    size_t sampleCount_ = 0, order_ = 0;
    bool solved_ = false, evalPending_ = false;
    double smoothing_ = 0.0, center_[3] = {}, scale_ = 1.0;
    std::string diagnostic_;
    cusolverDnHandle_t solver_ = nullptr;
    cudaEvent_t stateReady_ = nullptr, evalReady_ = nullptr; // cross-stream state/output ordering
    DeviceBuffer<float3> rest_, current_;
    DeviceBuffer<double> matrix_, work_, coefficients_, normSamples_;
    DeviceBuffer<double> gram_;
    // Device-side bind center/scale (cx, cy, cz, invScale): derived
    // inside the extent kernel so the direct bind's gram + matrix build
    // submit before the extent is host-proven and all three phases share
    // one synchronization. Bitwise the host's center_/scale_ derivation;
    // sized once, reused across binds.
    DeviceBuffer<double> bindParams_;
    DeviceBuffer<int> pivots_, evalFlags_;
    // Evaluate R cache, shared by the direct and fresh paths: the
    // radius-cubed kernel values are pose-invariant (they depend only on
    // the CVs, the rest samples, and the rest-derived center/scale), so a
    // verified cache turns the sqrt-bound evaluate into a streaming FMA
    // pass. rCvs_/rRest_ are the bitwise proof copies: a hit requires both
    // device memcmps to match, so no host trust and no caller versioning
    // is involved, and either path's fill serves the other. The direct
    // path verifies on the host between submits; the fresh path verifies
    // into rMiss_ and predicates its fill on the word, so the whole
    // verify/fill/eval sequence submits as one stream slice.
    DeviceBuffer<double> rCache_;
    DeviceBuffer<float3> rCvs_, rRest_;
    DeviceBuffer<int> rMiss_;
    int rN_ = 0;
    size_t rCount_ = 0;
    bool rValid_ = false;
    // Steady-state bind memo: rest is Default-time, so a sync-API caller
    // re-binds bitwise-identical samples every pose. A digest hit skips
    // the factor slice (rest copy, extent, gram, matrix build, LU) and
    // only re-establishes the identity state. The key is the FNV-1a
    // digest of the rest words (hashed on the host from a pinned staging
    // copy: a copy beats a kernel launch for kilobytes) plus the shape
    // and smoothing; a hit proves the factor buffers still hold this
    // bind's bytes, so every success-path byte is unchanged. Any slice
    // submit clears the memo (its writes land before success is known);
    // only a fully proven bind re-arms it.
    uint64_t bindDigest_ = 0;
    int bindMemoN_ = 0;
    double bindMemoSmoothing_ = 0.0;
    bool bindMemoValid_ = false;
    // Pinned staging for the bind-memo probe copy. Grown on demand,
    // freed with the binding; the digest copy is the only user, so no
    // reservation accounting applies.
    void* bindDigestStaging_ = nullptr;
    size_t bindDigestStagingBytes_ = 0;
    // Zero-copy direct-path proofs: one mapped allocation, read on the host
    // after the stream syncs, so no D2H node (and its ~7us drain bubble)
    // separates the phases. proofDev_ is re-queried if the binding moves.
    DirectProofs* proofHost_ = nullptr;
    DirectProofs* proofDev_ = nullptr;
    int proofDevice_ = -1;
    std::unique_ptr<FreshState> fresh_, acceptedFresh_, freshSolve_, freshEval_;
    std::unique_ptr<FreshState> retiredFresh_, retiredSolve_, retiredEval_;
    bool freshSolvePendingAcceptance_ = false;
};
}}
#endif
