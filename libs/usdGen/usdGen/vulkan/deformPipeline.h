// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// deformPipeline.h — Vulkan RBF deform pipeline.
//
// Ports the CPU reference in usdGen/deform/deformHost.{h,cpp} and the CUDA
// kernels in deformCuda.cu / rbf.cu to a two-submission GPU flow:
//
//   1. HOST: solve the RBF coefficients in double precision on the CPU
//      (deformRbfHost.h). A host rejection means the candidate is never
//      built.
//   2. EVALUATE (proof, synchronous): deformEvaluate.comp warps every control
//      vertex (CV) into a `warped` buffer and records a non-finite flag. Begin
//      submits it on a proof fence, waits, and reads the flag back — a bad
//      evaluation is rejected before any apply work is scheduled.
//   3. APPLY (candidate, asynchronous): deformApply.comp blends the input CVs
//      toward the warped CVs by an (optional) envelope with root-lock
//      correction, writing the output CVs. Poll() reports completion.
//
// Candidate semantics follow noisePipeline.h: proven by the evaluate step,
// pollable, quarantined on loss.

#pragma once

#include "usdGen/vulkan/chargedBuffer.h"
#include "usdGen/vulkan/deformRbfHost.h"

#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {

enum class DeformSemantic : uint32_t {
    Ok = 0,
    BadOffsets = 1,
    NonFinite = 2,
    BadValue = 3,
    RankDeficient = 4,
    SolverError = 5,
};

char const* DeformSemanticName(DeformSemantic semantic) noexcept;

// Optional cached-R evaluate SPIR-V (deformEvaluateVerify/Fill/Cached).
// Empty vectors disable the cache: the pipeline runs the direct shader
// exactly as before.
struct DeformEvalCacheSpirv {
    std::vector<uint32_t> verify;
    std::vector<uint32_t> fill;
    std::vector<uint32_t> cached;
};

// Test-only cached-evaluate seams. While disabled, Begin runs the direct
// shader and leaves the cache state untouched, so a test can compare the
// cached and direct paths bitwise on the same pipeline. The counters are
// global across pipelines; tests assert deltas. A hit evaluates through
// the cache without refilling, a miss refills it first (forced refills
// count as misses), a bypassed pose runs direct under miss backoff, and
// an unfunded pose runs direct for lack of budget. The funded prefix is
// the last R/proof capacity the cache funded (0 after an eviction).
void TestDisableDeformEvalCache(bool disable) noexcept;
uint64_t DeformEvalCacheHitsForTesting() noexcept;
uint64_t DeformEvalCacheMissesForTesting() noexcept;
uint64_t DeformEvalCacheBypassedForTesting() noexcept;
uint64_t DeformEvalCacheUnfundedForTesting() noexcept;
uint32_t DeformEvalCacheFundedPrefixForTesting() noexcept;

// Test-only heterogeneous-suffix seams. The force mode overrides the
// adaptive policy: 0 is automatic, 1 forces the host suffix, -1 forces the
// GPU suffix. Forcing hetero on an ineligible pose (no cache, no suffix,
// unreadable inputs) still runs the GPU suffix; the run counter tells
// which path ran. The counter is global across pipelines; tests assert
// deltas. An eligible forced-hetero pose always runs hetero (the tentative
// call reserves the mapping the final call confirms), so there is no
// fallback counter.
void TestForceDeformHeteroSuffix(int force) noexcept;
uint64_t DeformHeteroSuffixRunsForTesting() noexcept;

class DeformPipeline final : public std::enable_shared_from_this<DeformPipeline> {
public:
    class Candidate;
    using BeforeSubmit = std::function<bool()>;

    // Scalar field (mask). Domain: Groom=1 (global literal), Primitive=2
    // (per-curve), Point=4 (per-point). `data` null => `literal` applies to
    // every member of the domain and `count` is 0.
    struct ScalarField {
        float literal = 1.0f;
        uint32_t domain = 1;
        std::shared_ptr<const ChargedBuffer> data;
        uint32_t count = 0;
    };

    // Boolean field (enabled, lockRoots). `data` is uint8 (0/1). `enabled`
    // is groom-only; `lockRoots` is groom-or-primitive.
    struct BoolField {
        uint32_t literal = 0;
        uint32_t domain = 1;
        std::shared_ptr<const ChargedBuffer> data;
        uint32_t count = 0;
    };

    struct BeginInfo {
        std::shared_ptr<const ChargedBuffer> points;      // input CVs (3f each)
        std::shared_ptr<const ChargedBuffer> curveOffsets; // uint, [curves+1], monotone
        std::shared_ptr<const ChargedBuffer> rootTargets;  // per-curve root targets (3f each)
        uint32_t curveCount = 0;
        uint32_t pointCount = 0;

        // RBF sample pairs (rest, posed). restSamples[j] is a float3 index.
        // size() == 3*sampleCount; sampleCount in [4, 100].
        std::vector<float> restSamples;
        std::vector<float> posedSamples;
        int sampleCount = 0;
        double smoothing = 0.0;

        ScalarField mask;
        BoolField enabled;      // groom-only
        BoolField lockRoots;    // groom or primitive
        float groomEnvelope = 1.0f;
    };

    // `evaluateSpirv` / `applySpirv` are precompiled SPIR-V word arrays.
    // `cacheSpirv` arms the cached-R evaluate path; empty (the default)
    // keeps the direct-only pipeline.
    static std::shared_ptr<DeformPipeline> Create(
        std::shared_ptr<DeviceContext> context,
        std::vector<uint32_t> const& evaluateSpirv,
        std::vector<uint32_t> const& applySpirv,
        VkResult* result = nullptr,
        DeformEvalCacheSpirv const& cacheSpirv = {});

    ~DeformPipeline();

    DeformPipeline(DeformPipeline const&) = delete;
    DeformPipeline& operator=(DeformPipeline const&) = delete;

    std::shared_ptr<DeviceContext> const& context() const noexcept;

    // Host-validate, solve, record both dispatches; submit the evaluate proof
    // synchronously (waiting and reading its flag) and return a candidate that
    // is proven but not yet polled. `beforeSubmit` is invoked before the
    // evaluate submit. Rejects (null) on host error or a bad evaluate flag.
    std::unique_ptr<Candidate> Begin(
        BeginInfo info,
        VkResult* result = nullptr,
        DeformSemantic* semantic = nullptr,
        BeforeSubmit beforeSubmit = {});

private:
    struct Native;
    explicit DeformPipeline(std::shared_ptr<Native> native);
    std::shared_ptr<Native> native_;
};

class DeformPipeline::Candidate final {
public:
    struct Output {
        std::shared_ptr<const ChargedBuffer> points; // deformed CVs (3f each)
    };

    ~Candidate();

    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;

    // VK_NOT_READY: pending; VK_SUCCESS: done. On success `semantic` is
    // written and `output()` is valid. Never destroys the underlying work.
    VkResult Poll(DeformSemantic* semantic = nullptr);

    Output output() const noexcept;
    bool succeeded() const noexcept;

    // The input CV buffer the candidate was built from (identity for
    // cross-pool validation).
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;

    std::shared_ptr<DeviceContext> context() const noexcept;
    uint32_t pointCount() const noexcept;

    // Marks the candidate lost and releases GPU resources; idempotent.
    void Quarantine() noexcept;

private:
    struct State;
    explicit Candidate(std::shared_ptr<State> state);
    std::shared_ptr<State> state_;

    friend class DeformPipeline;
};

} // namespace usdGen::vulkan
