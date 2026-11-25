// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// usdGen — Vulkan expression application pipelines.
//
// ExprVkWidthApplyPipeline runs exprVkWidth.comp: the CUDA WidthKernel
// formula for a literal base, sampled profile, and literal-or-connected
// width/mask/replace/enabled. ExprVkExprOpPipeline runs exprVkExprOp.comp:
// the UsdGenExprOp::Evaluate displacement/width application under a literal
// or connected mask. Connected fields arrive as exprVkEvaluate 8-byte slots.
#ifndef USDGEN_VULKAN_EXPR_VK_APPLY_H
#define USDGEN_VULKAN_EXPR_VK_APPLY_H

#include "chargedBuffer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {

class ExprVkWidthApplyPipeline {
public:
    using BeforeSubmit = std::function<bool()>;
    static std::shared_ptr<ExprVkWidthApplyPipeline> Create(
        std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
        VkResult* result = nullptr);
    ~ExprVkWidthApplyPipeline();
    ExprVkWidthApplyPipeline(ExprVkWidthApplyPipeline const&) = delete;
    ExprVkWidthApplyPipeline& operator=(ExprVkWidthApplyPipeline const&) = delete;
    std::shared_ptr<DeviceContext> const& context() const noexcept;

    struct Controls {
        float widthLiteral = 0.01f;
        float maskLiteral = 1.0f;
        // 0 = literal, else 1 groom, 2 primitive, 4 point.
        uint32_t widthDom = 0, maskDom = 0, replaceDom = 0, enabledDom = 0;
        uint32_t replaceLiteral = 1, enabledLiteral = 1;
        float baseLiteral = 1.0f;
    };

    class Candidate {
    public:
        ~Candidate();
        Candidate(Candidate const&) = delete;
        Candidate& operator=(Candidate const&) = delete;
        void Quarantine() noexcept;
        bool succeeded() const noexcept;
        VkResult Poll(uint32_t* semanticStatus);
        VkResult Wait(uint64_t timeoutNs, uint32_t* semanticStatus);
        std::shared_ptr<const ChargedBuffer> output() const noexcept;
        std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
        uint32_t pointCount() const noexcept;
        uint32_t curveCount() const noexcept;
        std::shared_ptr<DeviceContext> context() const noexcept;
    private:
        friend class ExprVkWidthApplyPipeline;
        struct State;
        explicit Candidate(std::shared_ptr<State>);
        std::shared_ptr<State> state_;
    };

    // A null field selects the literal; a non-null field needs the dom.
    // Optional profile is 257 float32 samples. hairT is validated whenever
    // supplied; absent hairT uses ordinal coordinates from curveOffsets.
    // A nonempty sampled profile requires either hairT or curveOffsets.
    std::unique_ptr<Candidate> Begin(std::shared_ptr<const ChargedBuffer> input,
                                     std::shared_ptr<const ChargedBuffer> owners,
                                     std::shared_ptr<const ChargedBuffer> widthField,
                                     std::shared_ptr<const ChargedBuffer> maskField,
                                     std::shared_ptr<const ChargedBuffer> replaceField,
                                     std::shared_ptr<const ChargedBuffer> enabledField,
                                     uint32_t pointCount, uint32_t curveCount,
                                     Controls controls, VkResult* result = nullptr,
                                     BeforeSubmit beforeSubmit = {},
                                     std::shared_ptr<const ChargedBuffer> profile = {},
                                     std::shared_ptr<const ChargedBuffer> hairT = {},
                                     std::shared_ptr<const ChargedBuffer> curveOffsets = {});

private:
    struct Native;
    explicit ExprVkWidthApplyPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

class ExprVkExprOpPipeline {
public:
    using BeforeSubmit = std::function<bool()>;
    static std::shared_ptr<ExprVkExprOpPipeline> Create(
        std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
        VkResult* result = nullptr);
    ~ExprVkExprOpPipeline();
    ExprVkExprOpPipeline(ExprVkExprOpPipeline const&) = delete;
    ExprVkExprOpPipeline& operator=(ExprVkExprOpPipeline const&) = delete;
    std::shared_ptr<DeviceContext> const& context() const noexcept;

    class Candidate {
    public:
        ~Candidate();
        Candidate(Candidate const&) = delete;
        Candidate& operator=(Candidate const&) = delete;
        void Quarantine() noexcept;
        bool succeeded() const noexcept;
        VkResult Poll(uint32_t* semanticStatus);
        VkResult Wait(uint64_t timeoutNs, uint32_t* semanticStatus);
        std::shared_ptr<const ChargedBuffer> outPoints() const noexcept;
        std::shared_ptr<const ChargedBuffer> outWidths() const noexcept;
        std::shared_ptr<DeviceContext> context() const noexcept;
    private:
        friend class ExprVkExprOpPipeline;
        struct State;
        explicit Candidate(std::shared_ptr<State>);
        std::shared_ptr<State> state_;
    };

    // field holds count*components float32 slots (3 displacement, 1 width).
    // returnType 0 = displacement, 1 = width; perCurve selects primitive rate.
    std::unique_ptr<Candidate> Begin(std::shared_ptr<const ChargedBuffer> inPoints,
                                     std::shared_ptr<const ChargedBuffer> inWidths,
                                     std::shared_ptr<const ChargedBuffer> owners,
                                     std::shared_ptr<const ChargedBuffer> field,
                                     std::shared_ptr<const ChargedBuffer> maskField,
                                     uint32_t pointCount, uint32_t curveCount,
                                     uint32_t returnType, bool perCurve,
                                     uint32_t maskDom, float maskLiteral,
                                     VkResult* result = nullptr,
                                     BeforeSubmit beforeSubmit = {});

private:
    struct Native;
    explicit ExprVkExprOpPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_EXPR_VK_APPLY_H
