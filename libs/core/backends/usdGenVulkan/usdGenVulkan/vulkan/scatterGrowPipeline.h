// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_SCATTER_GROW_PIPELINE_H
#define USDGEN_VULKAN_SCATTER_GROW_PIPELINE_H

#include "chargedBuffer.h"

#include <cstdint>
#include <functional>
#include <vector>

namespace usdGen::vulkan {
// Queue-owner confined. Inputs must be immutable and have completed upload on
// this queue before Begin. Neither Begin nor Poll waits for device completion.
//
// ScatterGrow is the roots-only Grow producer: it takes captured scatter roots
// (positions, stableIds, rootPrim, rootUV, rootT/B/N) and emits fresh C3
// topology, one cvCount-CV curve per root, grown out along a deterministic
// per-curve target. There is no input curve topology, no rest plane, and no
// frame table to resolve -- roots are indexed directly by their ordinal.
//
// FP64 workaround: the device is created WITHOUT shaderFloat64, but the CUDA
// target math (scatterGrow.cu GrowKernel) is double-precision. The host
// precomputes each root's float target via BuildTargets (exact CUDA double
// formula) and Begin uploads it to the shader's read-only target plane. Begin
// therefore REQUIRES a complete precomputedTargets span in the Input and
// NEVER reads device stableIds on the host.
//
// pointCount is derived: curveCount * cvCount. It is not a separate Begin
// argument because the roots-only contract forbids any independent extent.
class ScatterGrowPipeline final
    : public std::enable_shared_from_this<ScatterGrowPipeline> {
public:
    class Candidate;
    // Owner-side admission checkpoint immediately before the first native
    // submit. It runs at most once only for a fully recorded nonempty
    // candidate; invalid input, allocation/recording failures, and
    // curves==0 never invoke it. True precedes pending state and every
    // native submit.
    using BeforeSubmit = std::function<bool()>;

    // Authored direction enum (ScatterGrowDirection mirror).
    enum class Direction : uint32_t { RootNormal = 0, RootTangent = 1, Literal = 2 };

    struct Controls {
        uint32_t cvCount = 2;
        int seed = 0;
        // CPU/CUDA Grow computes the random target in double then stores
        // float; these match ScatterGrowControls so BuildTargets is exactly
        // the CUDA double formula.
        double length = 1.0;
        double randomLo = 1.0, randomHi = 1.0;
        float lift = 0.0f;          // degrees about root B, in [-90,90]
        float fallbackWidth = 0.01f;
        Direction direction = Direction::RootNormal;
        float literalDirection[3] = {0.0f, 1.0f, 0.0f};
        // Root-normal rotation after lift, in [-360,360]; random in [0,1]
        // spans a full circle at 1 (ScatterGrowControls parity).
        float azimuth = 0.0f;
        float azimuthRandom = 0.0f;
    };

    // Roots-only geometry planes, all sized by curveCount (positions/rest are
    // 3*curveCount floats; ids packed u64; rootPrim int32; rootUV 2 floats).
    // Every buffer must originate from `context`, be proven complete uploads,
    // and cover the documented element count.
    struct Input {
        std::shared_ptr<const ChargedBuffer> positions; // float[3*curveCount]
        std::shared_ptr<const ChargedBuffer> stableIds; // u64[curveCount]
        std::shared_ptr<const ChargedBuffer> rootPrim;  // int32[curveCount]
        std::shared_ptr<const ChargedBuffer> rootUV;    // float[2*curveCount]
        std::shared_ptr<const ChargedBuffer> rootT;     // float[3*curveCount]
        std::shared_ptr<const ChargedBuffer> rootB;     // float[3*curveCount]
        std::shared_ptr<const ChargedBuffer> rootN;     // float[3*curveCount]
        std::shared_ptr<const ChargedBuffer> targets;   // float[curveCount], host-precomputed
    };

    // Exact CUDA double target math (scatterGrow.cu GrowKernel + DrawGrow +
    // Hash64): per id, r=DrawGrow(seed,id); target = float(length*(randomLo +
    // r*(randomHi-randomLo))). Each draw uses the high 24 bits of the
    // SplitMix64 finalizer output, matching UsdGenDraw01/hash.h. Called from
    // the executor's HOST capture path, where stableIds are host-resident.
    static std::vector<float> BuildTargets(std::vector<uint64_t> const& stableIds,
                                           Controls const& controls);

    // CPU-fallback geometry producer: BuildTargets + the host equivalent of
    // GrowKernel (offsets, finite scans, direction normalize + lift, per-CV
    // resample, width fallback, hairT).
    struct Outputs {
        std::vector<float> points, rest;      // 3*curveCount*cvCount each
        std::vector<float> widths, hairT;     // curveCount*cvCount each
        std::vector<uint32_t> offsets;        // curveCount+1
        std::vector<uint64_t> ids;            // curveCount
        std::vector<int32_t> rootPrim;        // curveCount
        std::vector<float> rootUV;            // 2*curveCount
        std::vector<float> rootT, rootB, rootN; // 3*curveCount each
    };
    // Returns false and leaves `output` untouched on any semantic rejection
    // (matching the CUDA error codes: NonFinite).
    static bool BuildCpu(std::vector<float> const& positions,
                         std::vector<uint64_t> const& stableIds,
                         std::vector<int32_t> const& rootPrim,
                         std::vector<float> const& rootUV,
                         std::vector<float> const& rootT,
                         std::vector<float> const& rootB,
                         std::vector<float> const& rootN,
                         std::vector<float> const& targets,
                         Controls const& controls, Outputs* output);

    // SPIR-V must be the trusted, externally spirv-val-validated
    // scatterGrow.comp module with the documented binding/48-byte
    // push-constant ABI (12 words).
    static std::shared_ptr<ScatterGrowPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& spirv,
        VkResult* result = nullptr);
    ~ScatterGrowPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;

    // `curveCount` is the input root extent; outputs cover curveCount*cvCount
    // points. `inputOwner` keeps every borrowed input alive through terminal
    // proof.
    std::unique_ptr<Candidate> Begin(std::shared_ptr<const void> inputOwner,
        Input const& input, uint32_t curveCount,
        Controls const& controls, VkResult* result = nullptr, BeforeSubmit = {});
private:
    struct Native;
    explicit ScatterGrowPipeline(std::shared_ptr<Native>);
    std::unique_ptr<Candidate> BeginInternal(std::shared_ptr<const void>,
        Input const&, uint32_t, Controls const&, VkResult*, BeforeSubmit);
    std::shared_ptr<Native> native_;
};

class ScatterGrowPipeline::Candidate final {
public:
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;
    // VK_SUCCESS means device proof obtained; semanticStatus must additionally
    // be kOk before geometry is exposed. VK_NOT_READY preserves the candidate.
    // semanticStatus: 0 Ok, 2 NonFinite.
    enum class Semantic : uint32_t { Ok = 0, NonFinite = 2 };
    VkResult Poll(Semantic* semantic = nullptr);
    // Primary output (points plane); all planes are exposed once proved.
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    uint32_t curveCount() const noexcept;
    uint32_t pointCount() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    void Quarantine() noexcept;
    std::shared_ptr<const ChargedBuffer> rest() const noexcept;
    std::shared_ptr<const ChargedBuffer> widths() const noexcept;
    std::shared_ptr<const ChargedBuffer> hairT() const noexcept;
    std::shared_ptr<const ChargedBuffer> offsets() const noexcept;
    std::shared_ptr<const ChargedBuffer> ids() const noexcept;
    std::shared_ptr<const ChargedBuffer> rootPrim() const noexcept;
    std::shared_ptr<const ChargedBuffer> rootUV() const noexcept;
    std::shared_ptr<const ChargedBuffer> rootT() const noexcept;
    std::shared_ptr<const ChargedBuffer> rootB() const noexcept;
    std::shared_ptr<const ChargedBuffer> rootN() const noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class ScatterGrowPipeline;
};
} // namespace usdGen::vulkan
#endif
