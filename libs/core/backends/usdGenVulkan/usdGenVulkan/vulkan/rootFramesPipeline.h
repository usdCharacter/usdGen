// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_ROOT_FRAMES_PIPELINE_H
#define USDGEN_VULKAN_ROOT_FRAMES_PIPELINE_H

#include "chargedBuffer.h"
#include <functional>
#include <vector>

namespace usdGen::vulkan {

// Normal-domain selector for authored frame normals; mirrors CUDA.
// None=0 Constant=1 Uniform=2 Vertex=3 FaceVarying=4.
enum class RootFrameNormalDomain : uint32_t {
    None = 0,
    Constant = 1,
    Uniform = 2,
    Vertex = 3,
    FaceVarying = 4,
};

// Mirror of CUDA rootFrames status codes. NOTE: there is NO kMissing; the
// codes are asserted by tests, so this mapping is authoritative.
enum class RootFramesBeginStatus : uint32_t {
    Ok = 0,
    BadShape = 1,        // kBadShape
    NonFinite = 2,       // kNonFinite
    BadTopology = 3,     // kBadTopology
    BadRoot = 4,         // kBadRoot
    BadAuthoredFrame = 5 // kBadAuthoredFrame (host/authored-path only)
};

// Queue-owner confined. Inputs must be immutable and have completed upload on
// this queue before Begin. Neither Begin nor Poll waits for device completion
// beyond Begin's own synchronous validation-readback step.
class RootFramesPipeline final : public std::enable_shared_from_this<RootFramesPipeline> {
public:
    class Candidate;

    // Owner-side admission checkpoint immediately before the first native
    // submit. Runs once only for a fully recorded candidate; invalid input,
    // allocation/recording failures, semantic rejection, and curves==0 never
    // invoke it. True precedes pending state and every native submit. False
    // or an exception rejects without arming or submitting the candidate.
    using BeforeSubmit = std::function<bool()>;

    // Everything needed to begin a root-frame build. `context` is implicit
    // from `restVerts->context()` (the first buffer).
    struct BeginInfo {
        std::shared_ptr<const ChargedBuffer> restVerts;   // float[3*vertexCount]
        std::shared_ptr<const ChargedBuffer> faceOffsets; // uint[faceCount+1]
        std::shared_ptr<const ChargedBuffer> faceIndices; // uint[indexCount]
        std::shared_ptr<const ChargedBuffer> rootPrim;    // int[curves]
        std::shared_ptr<const ChargedBuffer> rootUV;      // float[2*curves]
        std::shared_ptr<const ChargedBuffer> normals;     // float[3*...] or null when domain None
        RootFrameNormalDomain normalDomain = RootFrameNormalDomain::None;
        uint32_t vertexCount = 0;
        uint32_t faceCount = 0;
        uint32_t indexCount = 0;
        uint32_t curves = 0;
        // Row-major 16-double authored matrices, one per curve (curves*16).
        // authoredFrames==nullptr (default) selects the device build path;
        // non-null selects the host authored path (skips the shader entirely).
        // (C++17: pointer+count; std::span is C++20 in this toolchain.)
        double const* authoredFrames = nullptr;
        uint32_t authoredFrameCount = 0; // number of doubles (curves*16)
    };

    // Both SPIR-V modules must be trusted, externally spirv-val-validated
    // rootFramesValidate.comp (validate) and rootFrames.comp (build) with the
    // documented binding/push-constant ABIs.
    static std::shared_ptr<RootFramesPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& validateSpirv,
        std::vector<uint32_t> const& buildSpirv, VkResult* result = nullptr);
    ~RootFramesPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;

    // Begin performs host-side shape validation, the host authored-frame
    // validation sweep (when authoredFrames non-empty), and for the device
    // path a SYNCHRONOUS validated dispatch (fill status, validate) with
    // readback. On semantic failure returns nullptr and sets `beginStatus`
    // (no candidate, no outputs — matches CUDA failure-with-no-outputs). On
    // success returns a candidate whose recorded build dispatch is submitted
    // asynchronously; candidate->Poll() reads status + drop count.
    std::unique_ptr<Candidate> Begin(BeginInfo info, VkResult* result = nullptr,
        RootFramesBeginStatus* beginStatus = nullptr, BeforeSubmit beforeSubmit = {});

private:
    struct Native;
    explicit RootFramesPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

// One candidate owns a single root-frame build for `curves` curves. It holds
// every input buffer (inputOwner is the first/context bearer) so the whole
// dependency graph stays alive until Poll or destruction.
class RootFramesPipeline::Candidate final {
public:
    // Result planes: flat scalar float[3*curves] storage buffers plus the
    // u32-lane valid/drop arrays. All six plane owners are always present.
    struct Output {
        std::shared_ptr<const ChargedBuffer> origin;
        std::shared_ptr<const ChargedBuffer> tangent;
        std::shared_ptr<const ChargedBuffer> binormal;
        std::shared_ptr<const ChargedBuffer> normal;
        std::shared_ptr<const ChargedBuffer> valid; // u32[curves] 0/1
        std::shared_ptr<const ChargedBuffer> drop;  // u32[curves] 0/1
    };

    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;

    // VK_SUCCESS means device proof obtained; `semanticStatus` must be zero
    // before output is exposed. `badRootCount` receives the number of dropped
    // curves (device path readback; authored path writes zero). VK_NOT_READY
    // preserves the candidate.
    VkResult Poll(uint32_t* semanticStatus = nullptr, uint32_t* badRootCount = nullptr);
    // Exposes the output planes only when proved and semantically clean.
    Output output() const noexcept;
    bool succeeded() const noexcept;
    // Context-bearing first input (restVerts); all inputs stay alive via the
    // candidate's internal state regardless.
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    uint32_t count() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    void Quarantine() noexcept;

private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class RootFramesPipeline;
};

// Convenience: strings for RootFramesBeginStatus (for diagnostics).
char const* RootFramesBeginStatusName(RootFramesBeginStatus) noexcept;

} // namespace usdGen::vulkan
#endif