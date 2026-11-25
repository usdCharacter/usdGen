// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_POINT_OVERRIDE_PIPELINE_H
#define USDGEN_VULKAN_POINT_OVERRIDE_PIPELINE_H

#include "chargedBuffer.h"
#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {
// Queue-owner confined. Applies a sparse, indexed absolute point edit to a
// borrowed device base, publishing a private full output buffer only after
// every validation phase is clean (strict-ORDER parity with CudaPointOverride:
// no partial writes ever). Mirrors WidthPipeline's lifecycle
// (Create / Begin -> unique_ptr<Candidate> / Poll / succeeded / output /
// inputOwner / context). Unlike WidthPipeline, Begin performs a SYNCHRONOUS
// readback of the validation phases before recording the scatter phase; this
// is the documented structural deviation that preserves CUDA's early-return
// fidelity (see integration-topology.md).
class PointOverridePipeline final : public std::enable_shared_from_this<PointOverridePipeline> {
public:
    class Candidate;
    // CUDA-equivalent semantic statuses (PointOverrideStatus ordinal mapping).
    enum class Semantic : uint32_t { Ok = 0, InvalidArgument = 1, NonFinite = 2, Duplicate = 3 };
    using BeforeSubmit = std::function<bool()>;
    // SPIR-V must be the trusted, externally spirv-val-validated
    // pointOverride.comp module (single "main" entry, phase selector in the
    // push constant) with the six-binding/12-byte push-constant ABI:
    // b0 base float[3*baseN], b1 indices int32[slotN], b2 replacements
    // float[3*slotN], b3 outputPlane float[3*baseN], b4 error uint, b5
    // occupancy uint[baseN].
    static std::shared_ptr<PointOverridePipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& spirv,
        VkResult* result = nullptr);
    ~PointOverridePipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    // Host Begin pre-checks: indices.size == replacements.size (== slotN);
    // sizes <= INT32_MAX; slotN <= baseN; pointer/usage consistency and any
    // input must belong to context and be unproven-free. The base is borrowed
    // and never modified; output is a private full-point buffer.
    std::unique_ptr<Candidate> Begin(
        std::shared_ptr<const ChargedBuffer> base,
        std::shared_ptr<const ChargedBuffer> indices,
        std::shared_ptr<const ChargedBuffer> replacements,
        uint32_t baseN, uint32_t slotN,
        VkResult* result = nullptr, BeforeSubmit beforeSubmit = {});
private:
    struct Native;
    explicit PointOverridePipeline(std::shared_ptr<Native>);
    std::unique_ptr<Candidate> BeginInternal(
        std::shared_ptr<const ChargedBuffer>, std::shared_ptr<const ChargedBuffer>,
        std::shared_ptr<const ChargedBuffer>, uint32_t, uint32_t,
        VkResult*, BeforeSubmit);
    std::shared_ptr<Native> native_;
};

class PointOverridePipeline::Candidate final {
public:
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;
    // VK_SUCCESS means device proof obtained. *status carries the Semantic
    // (Ok when the edit is publishable). VK_NOT_READY preserves the candidate.
    VkResult Poll(Semantic* status = nullptr);
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    void Quarantine() noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class PointOverridePipeline;
};
} // namespace usdGen::vulkan
#endif // USDGEN_VULKAN_POINT_OVERRIDE_PIPELINE_H