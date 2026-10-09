// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// rbfVkBinding.h — Vulkan port of the CudaRbfBinding device path
// (libs/usdGen/usdGen/gpu/rbf.cu).
//
// The admitted plan range rbfSamples [4,100] is host-solved
// (vulkan/deformRbfHost.h); this binding solves on the device so the full
// gap range executes: budgets above 100 (up to kRbfVkMaxSamples),
// below-CUDA-minimum counts (which fail at Bind with the exact CUDA
// InvalidArgument result instead of a plan rejection), and baked
// non-identity sample sets (rbfVkPlan.h poses them in source-local
// space; the solver itself is transform-agnostic, like cuSOLVER).
//
// Phase mapping to rbf.cu: extentKernel -> rbfVkExtent.comp (+ host
// center/scale, as in Bind), polynomialGram -> rbfVkGram.comp (+ host
// fullAffineRank, as in Bind), buildMatrix -> rbfVkBuildMatrix.comp,
// cusolverDnDgetrf -> rbfVkLu.comp, rhsKernel/zeroTail -> rbfVkRhs.comp,
// cusolverDnDgetrs -> rbfVkTriSolve.comp, evalKernel -> rbfVkEvaluate.comp.
// The CUDA direct-evaluate R cache (rFillKernel + evalCachedKernel +
// verifyKernel) maps to rbfVkFill.comp + rbfVkEvaluateCached.comp +
// rbfVkVerify.comp under RbfVkEvalCache, with the deform pipeline's
// single-submit verify/snapshot/conditional-fill protocol (no host
// round-trip between verify and fill).
// Scalar diagnostics (extent, gram, info, flags) are read back after each
// fenced submit, mirroring the CUDA stream-synchronized queries.
//
// Status codes and diagnostic strings mirror RbfStatus in gpu/rbf.h
// verbatim (DeviceError is the CudaError analog); the parity test asserts
// equal strings on the shared error paths. Production uses BeginBind,
// AdvanceBind, BeginSolve and nonblocking Poll methods. Bind/Solve are synchronous
// (fence-waited, like the CUDA stream-synchronized originals); Evaluate
// is asynchronous and Finish() completes it. Owner-confined: no internal
// locking, like CudaRbfBinding.
#pragma once

#include "usdGen/vulkan/chargedBuffer.h"
#include "usdGen/vulkan/deviceContext.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace usdGen::vulkan {

// Shared CUDA/Vulkan signed matrix-index safety bound: (n+4)^2 <= INT_MAX.
// Actual Vulkan admission additionally checks the selected device storage
// range and charged memory budget. Create allocates no sample storage;
// Bind sizes its buffers to the requested count after proving prior work.
inline constexpr int kRbfVkMaxSamples = 46336;

enum class RbfVkStatus {
    Ok,
    InvalidArgument,
    NonFiniteInput,
    RankDeficient,
    DeviceError, // VkError analog of RbfStatus::CudaError
    SolverError,
};

char const* RbfVkStatusName(RbfVkStatus status) noexcept;

// Persistent LU-factor cache shared across RbfVkBindings on one device.
// Rest samples are static across poses while only the posed samples move,
// so a gap-range groom re-factors an identical matrix on every pose
// (41ms on GB10 for n=400). A hit adopts the stored factors and skips
// the gram + buildMatrix + LU submit and its readbacks; the Bind/Poll
// protocol, admission hooks, and buffer state are unchanged. Only
// successful binds are stored (failures re-run and fail as before), and
// the key covers the device, count, smoothing bits, and rest bytes, so a
// hit adopts bitwise what a fresh bind would compute. Thread-safe;
// single-entry (one rest at a time, like the CPU boundRest_ skip).
class RbfVkFactorCache {
public:
    RbfVkFactorCache() = default;
    RbfVkFactorCache(RbfVkFactorCache const&) = delete;
    RbfVkFactorCache& operator=(RbfVkFactorCache const&) = delete;

    struct Entry {
        DeviceContext* context = nullptr;
        int n = 0, m = 0;
        uint64_t smoothingBits = 0;
        std::vector<float> rest;
        std::vector<double> lu;
        std::vector<int> perm;
    };

private:
    friend class RbfVkBinding;
    std::shared_ptr<Entry const> Lookup(DeviceContext* context, int n,
                                        double smoothing, float const* rest);
    void Store(DeviceContext* context, int n, double smoothing,
               std::vector<float> const& rest, std::vector<double> const& lu,
               std::vector<int> const& perm, int m);
    std::mutex mutex_;
    std::shared_ptr<Entry const> entry_;
};

struct RbfVkBindingSpirv {
    std::vector<uint32_t> extent;
    std::vector<uint32_t> gram;
    std::vector<uint32_t> buildMatrix;
    std::vector<uint32_t> lu;
    std::vector<uint32_t> rhs;
    std::vector<uint32_t> triSolve;
    std::vector<uint32_t> evaluate;
    // Optional R-cache programs (all three or none): without them the
    // binding evaluates directly exactly as before. With them and an
    // eval cache set, repeated evaluations over unchanged CVs/rest run
    // through the cached radii instead of the fp64 sqrt loop.
    std::vector<uint32_t> verify;
    std::vector<uint32_t> fill;
    std::vector<uint32_t> evaluateCached;
};

// Pose-invariant R cache shared across RbfVkBindings on one device. R
// depends only on the CVs, the rest samples, and the rest-derived
// center/scale, all verified bitwise (CVs on the device, rest/n on the
// host), so a hit evaluates through the cache and a miss refills it
// first; either way the bytes match the direct shader bit for bit. The
// whole cached submit (decide, submit, wait, bookkeeping) runs under
// one mutex, so concurrent evaluations sharing a cache serialize like
// the deform pipeline's proof phase. Single-entry (one shape at a
// time); over the cap, or when a reshape does not fit the pool, the
// binding evaluates directly with no added work. Thread-safe.
class RbfVkEvalCache {
public:
    RbfVkEvalCache() = default;
    RbfVkEvalCache(RbfVkEvalCache const&) = delete;
    RbfVkEvalCache& operator=(RbfVkEvalCache const&) = delete;

    // True outcomes from the indirect-args readback (a speculative
    // evaluation the device refilled counts a miss). For tests.
    uint64_t hitsForTesting() const;
    uint64_t missesForTesting() const;

private:
    friend class RbfVkBinding;
    mutable std::mutex mutex_;
    DeviceContext* context_ = nullptr;
    int entryN_ = 0;
    uint32_t entryP_ = 0;
    double entryCenter_[3] = {};
    double entryScale_ = 1.0;
    std::vector<float> entryRest_;
    bool armed_ = false;
    std::shared_ptr<ChargedBuffer> rCache_;
    std::shared_ptr<ChargedBuffer> proof_;
    std::shared_ptr<ChargedBuffer> fillArgs_;
    std::shared_ptr<ChargedBuffer> cachedArgs_;
    std::shared_ptr<ChargedBuffer> verifyUbo_;
    uint64_t hits_ = 0;
    uint64_t misses_ = 0;
};

// Global test seam (mirrors CUDA's TestDisableCudaRbfEvalCache): while
// set, every RbfVkBinding evaluates directly even with a cache set.
void TestDisableRbfVkEvalCache(bool disable) noexcept;

class RbfVkBinding final : public std::enable_shared_from_this<RbfVkBinding> {
public:
    static std::shared_ptr<RbfVkBinding> Create(
        std::shared_ptr<DeviceContext> context,
        RbfVkBindingSpirv const& spirv,
        VkResult* result = nullptr);

    ~RbfVkBinding();

    RbfVkBinding(RbfVkBinding const&) = delete;
    RbfVkBinding& operator=(RbfVkBinding const&) = delete;

    std::shared_ptr<DeviceContext> const& context() const noexcept;

    using BeforeSubmit = std::function<bool()>;
    // Owner-driven production path: one submit per Begin/Advance and Poll
    // consumes only an exact fence proof. No Vulkan wait occurs here.
    // BeginSolve is staged, not submitted: it stashes the pose and runs
    // the admission hook, and the following PollSolve runs the solve on
    // the host (bitwise the retired rhs + triSolve submits) with no fence.
    // BeginBind is likewise staged: PollSolve runs the extent on the host
    // (exact min/max port of the retired extent submit) with no fence,
    // while AdvanceBind still submits the gram/matrix/LU. The pending
    // protocol is unchanged: a staged phase reports pending until
    // PollSolve consumes it, exactly like a fenced submit.
    RbfVkStatus BeginBind(float const*, int, double, BeforeSubmit = {});
    RbfVkStatus AdvanceBind(BeforeSubmit = {});
    RbfVkStatus BeginSolve(float const*, int, BeforeSubmit = {});
    VkResult PollSolve(RbfVkStatus* = nullptr);
    VkResult PollEvaluate(RbfVkStatus* = nullptr);
    bool HasPendingSolve() const noexcept;
    VkResult lastResult() const noexcept;

    // Uploads rest samples, runs the extent on the host and the
    // gram/matrix/LU on the device, and establishes the identity
    // (zero-coefficient) solved state. Mirrors CudaRbfBinding::Bind,
    // including the n<4 InvalidArgument result.
    RbfVkStatus Bind(float const* restSamples, int sampleCount, double smoothing);

    // Uploads posed samples, builds the RHS and triangular-solves on the
    // device. `posedCount` must equal the bound sample count, mirroring
    // CudaRbfBinding::Solve's view-size check.
    RbfVkStatus Solve(float const* posedSamples, int posedCount);

    // Asynchronous device evaluation into `out` (both 3*count floats).
    // Stacking is allowed like CUDA (one shared completion diagnostic);
    // Finish() completes all pending evaluations.
    RbfVkStatus Evaluate(std::shared_ptr<const ChargedBuffer> cvs,
                         std::shared_ptr<ChargedBuffer> out,
                         uint32_t count, BeforeSubmit beforeSubmit = {});
    RbfVkStatus Finish();

    // Synchronous host convenience: upload, Evaluate, Finish, read back.
    RbfVkStatus EvaluateHost(float const* cvs, float* out, uint32_t count);

    bool HasPendingEvaluate() const noexcept;
    int sampleCount() const noexcept;
    int order() const noexcept; // sampleCount + 4, or 0 when unbound
    double center(int axis) const noexcept;
    double scale() const noexcept;
    char const* diagnostic() const noexcept;
    // Optional factor cache (null clears). The cache must outlive the
    // binding's binds; entries are device-specific and immutable.
    void SetFactorCache(std::shared_ptr<RbfVkFactorCache> cache);
    // Optional R cache (null clears). The cache must outlive the
    // binding's evaluations; cached evaluates retain the entry buffers
    // for the submit lifetime either way.
    void SetEvalCache(std::shared_ptr<RbfVkEvalCache> cache);

private:
    struct Native;
    explicit RbfVkBinding(std::shared_ptr<Native> native);
    RbfVkStatus fail(RbfVkStatus status, char const* what);
    // Cached-path Evaluate: runs the verify/fill/cached submit and
    // returns true with *status set, or returns false to run the direct
    // shader with no added work. The caller has validated and waited
    // for stacked work; the cached submit waits for its own fence (the
    // deform proof-phase discipline, serializing shared-cache users)
    // but keeps the async protocol (evalPending_/Finish/PollEvaluate
    // behave as if the submit were still in flight).
    bool EvaluateCached(std::shared_ptr<const ChargedBuffer> cvs,
                        std::shared_ptr<ChargedBuffer> out, uint32_t count,
                        BeforeSubmit beforeSubmit,
                        VkPhysicalDeviceProperties const& physical,
                        RbfVkStatus* status);
    std::shared_ptr<Native> native_;
    std::shared_ptr<RbfVkFactorCache> factorCache_;
    std::shared_ptr<RbfVkEvalCache> evalCache_;
    int sampleCount_ = 0, order_ = 0;
    bool solved_ = false;
    bool evalPending_ = false;
    double center_[3] = {};
    double scale_ = 1.0;
    std::string diagnostic_;
};

} // namespace usdGen::vulkan
