// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_STYLE_VK_PIPELINE_H
#define USDGEN_VULKAN_STYLE_VK_PIPELINE_H

#include "chargedBuffer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {

// Vulkan twin of the CUDA device-only style primitives (gpu/styleOps.h).
// One pipeline object serves the three finished operators with per-op
// dispatch (BeginWidthRamp/BeginLength/BeginGrow); Noise has no lowering on
// either backend and reports NotSupported on both.
//
// Contract mapping to CudaStyleOps:
//   Begin*  == WidthRamp/Length/Grow queue call (host validation only;
//             device diagnostics are deferred, like CUDA returning Ok and
//             diagnosing at Finish).
//   Poll    == Finish (device diagnostic readback; the output plane is
//             exposed only when the diagnostic is clean, like CUDA
//             publishing its staging buffer only on success).
//   ScalarField domains reuse the NoisePipeline codes (Groom=1,
//             Primitive=2, Point=4); a literal field must use Groom, exactly
//             like CUDA validateField.
//
// Style primitives have no plan surface: no executor, compiler, or plan
// lowering routes through CudaStyleOps (only tests consume it), so this
// pipeline needs no executionPlan hook. CUDA equivalence is proven by the
// device parity test, not by plan acceptance.

// Device diagnostic codes. Mirror CUDA styleOps.cu exactly: 0 clean,
// 1 kBadOffsets, 2 kNonFinite, 3 kBadValue.
enum class StyleVkSemantic : uint32_t {
    Ok = 0,
    BadOffsets = 1,
    NonFinite = 2,
    BadValue = 3,
};

char const* StyleVkSemanticName(StyleVkSemantic) noexcept;

// Host-call status. Mirrors CudaStyleOps StyleStatus; VulkanError replaces
// CudaError (a native failure, never a silent degradation).
enum class StyleVkStatus {
    Ok,
    InvalidArgument,
    NonFiniteInput,
    InvalidValue,
    NotSupported,
    VulkanError,
};

char const* StyleVkStatusName(StyleVkStatus) noexcept;
// DecodeError mirror: Ok -> Ok, BadOffsets -> InvalidArgument, NonFinite ->
// NonFiniteInput, BadValue -> InvalidValue. Any other code maps to
// VulkanError.
StyleVkStatus StyleVkStatusForSemantic(StyleVkSemantic) noexcept;

// Scalar field codes shared with NoisePipeline.
struct StyleVkDomain {
    static constexpr uint32_t Groom = 1u;
    static constexpr uint32_t Primitive = 2u;
    static constexpr uint32_t Point = 4u;
};

// Queue-owner confined. Inputs must be immutable and have completed upload
// on this queue before Begin. Neither Begin nor Poll waits for device
// completion: Begin records one submit (validate phase + compute phase, in
// CUDA stream order) and Poll proves it.
class StyleVkPipeline final : public std::enable_shared_from_this<StyleVkPipeline> {
public:
    class Candidate;

    // Owner-side admission checkpoint immediately before the single native
    // submit. Runs once only for a fully recorded candidate; host-invalid
    // input, allocation/recording failures, and empty geometry (which
    // submits nothing) never invoke it.
    using BeforeSubmit = std::function<bool()>;

    // A scalar (float) field parameter. Either a groom literal or a device
    // data plane in one of the three domains. Mirrors gpu::ScalarField.
    struct ScalarField {
        float literal = 0.0f;
        uint32_t domain = StyleVkDomain::Groom;
        std::shared_ptr<const ChargedBuffer> data;
        uint32_t count = 0;
    };

    struct WidthRampInfo {
        std::shared_ptr<const ChargedBuffer> offsets; // uint[curveCount+1]
        uint32_t curveCount = 0;
        uint32_t pointCount = 0;
        ScalarField root;
        ScalarField tip;
        // Optional hairT (float[pointCount]); null selects inclusive
        // CV-index interpolation, like CUDA's omitted hairT.
        std::shared_ptr<const ChargedBuffer> hairT;
    };

    struct LengthInfo {
        std::shared_ptr<const ChargedBuffer> points;  // float[3*pointCount]
        std::shared_ptr<const ChargedBuffer> offsets; // uint[curveCount+1]
        uint32_t curveCount = 0;
        uint32_t pointCount = 0;
        ScalarField scale;
    };

    struct GrowInfo {
        std::shared_ptr<const ChargedBuffer> offsets; // uint[curveCount+1]
        std::shared_ptr<const ChargedBuffer> roots;   // float[3*curveCount]
        std::shared_ptr<const ChargedBuffer> normals; // float[3*curveCount]
        uint32_t curveCount = 0;
        uint32_t pointCount = 0;
        ScalarField length;
        // Optional hairT (float[pointCount]); null selects inclusive
        // CV-index interpolation, like CUDA's omitted hairT.
        std::shared_ptr<const ChargedBuffer> hairT;
    };

    // All three SPIR-V modules must be trusted, externally spirv-val-
    // validated styleVkWidthRamp.comp / styleVkLength.comp /
    // styleVkGrow.comp with the documented binding/push-constant ABIs.
    static std::shared_ptr<StyleVkPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& widthRampSpirv,
        std::vector<uint32_t> const& lengthSpirv,
        std::vector<uint32_t> const& growSpirv, VkResult* result = nullptr);
    // Noise is deliberately not implemented, exactly like
    // CudaStyleOps::Noise which reports NotSupported: the complete planned
    // operator needs explicit rest-noise base, frame, hairT, and exact FBM
    // controls that neither backend wires through this surface.
    static StyleVkStatus NoiseStatus() noexcept { return StyleVkStatus::NotSupported; }
    ~StyleVkPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;

    // Each Begin performs host-side validation (mirroring CUDA begin() plus
    // validateField) and records the op. On host-invalid input returns
    // nullptr with *status set (InvalidArgument/NonFiniteInput) and *result
    // set to VK_ERROR_INITIALIZATION_FAILED; on native failure returns
    // nullptr with *status VulkanError. Device diagnostics surface through
    // Candidate::Poll, like CUDA Finish.
    std::unique_ptr<Candidate> BeginWidthRamp(WidthRampInfo info,
        VkResult* result = nullptr, StyleVkStatus* status = nullptr,
        BeforeSubmit beforeSubmit = {});
    std::unique_ptr<Candidate> BeginLength(LengthInfo info,
        VkResult* result = nullptr, StyleVkStatus* status = nullptr,
        BeforeSubmit beforeSubmit = {});
    std::unique_ptr<Candidate> BeginGrow(GrowInfo info,
        VkResult* result = nullptr, StyleVkStatus* status = nullptr,
        BeforeSubmit beforeSubmit = {});

private:
    struct Native;
    struct OpNative;
    explicit StyleVkPipeline(std::shared_ptr<Native>);
    std::unique_ptr<Candidate> BeginImpl(OpNative const* op,
        std::shared_ptr<const ChargedBuffer> const* bindings, uint32_t bindingCount,
        void const* push, uint32_t pushBytes, uint32_t curveCount,
        uint32_t pointCount, bool vectorOutput, VkResult* result,
        StyleVkStatus* status, BeforeSubmit beforeSubmit,
        std::shared_ptr<const ChargedBuffer> const* owners, uint32_t ownerCount);
    std::shared_ptr<Native> native_;
};

// One candidate owns a single style operation. It holds every input buffer
// so the whole dependency graph stays alive until Poll or destruction.
class StyleVkPipeline::Candidate final {
public:
    // Result plane: float[pointCount] for WidthRamp, float[3*pointCount]
    // for Length/Grow. Empty geometry yields a 4-byte dummy plane with
    // pointCount() == 0 (zero bytes are meaningful).
    struct Output {
        std::shared_ptr<const ChargedBuffer> values;
    };

    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;

    // VK_SUCCESS means device proof obtained; `semantic` must be Ok before
    // output is exposed. VK_NOT_READY preserves the candidate.
    VkResult Poll(StyleVkSemantic* semantic = nullptr);
    // Exposes the output plane only when proved and semantically clean.
    Output output() const noexcept;
    bool succeeded() const noexcept;
    bool vectorOutput() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    uint32_t pointCount() const noexcept;
    uint32_t curveCount() const noexcept;
    void Quarantine() noexcept;

private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class StyleVkPipeline;
};

} // namespace usdGen::vulkan

#endif
