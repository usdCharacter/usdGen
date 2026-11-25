// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// usdGen — Vulkan expression evaluator pipeline (exprVkEvaluate.comp).
//
// One dispatch runs one exact IR prefix over count elements into 8-byte
// output slots (see exprVkEvaluate.comp). The caller owns the program, field
// and owners buffers; this owns the output and status buffers. Status 0 is
// success, 1 is InvalidValue (a cooked value the destination cannot
// represent); raw segments cannot fail validation.
#ifndef USDGEN_VULKAN_EXPR_VK_EVALUATE_H
#define USDGEN_VULKAN_EXPR_VK_EVALUATE_H

#include "chargedBuffer.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {

class ExprVkEvaluatePipeline {
public:
    using BeforeSubmit = std::function<bool()>;
    static std::shared_ptr<ExprVkEvaluatePipeline> Create(
        std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
        VkResult* result = nullptr);
    ~ExprVkEvaluatePipeline();
    ExprVkEvaluatePipeline(ExprVkEvaluatePipeline const&) = delete;
    ExprVkEvaluatePipeline& operator=(ExprVkEvaluatePipeline const&) = delete;
    std::shared_ptr<DeviceContext> const& context() const noexcept;

    class Candidate {
    public:
        ~Candidate();
        Candidate(Candidate const&) = delete;
        Candidate& operator=(Candidate const&) = delete;
        void Quarantine() noexcept;
        bool succeeded() const noexcept;
        // Nonblocking proof; writes the semantic status (0 ok, 1 invalid).
        VkResult Poll(uint32_t* semanticStatus);
        // Blocking proof for synchronous orchestration (mirrors the CUDA
        // Finish fence wait); a timeout leaves the candidate pending.
        VkResult Wait(uint64_t timeoutNs, uint32_t* semanticStatus);
        std::shared_ptr<const ChargedBuffer> output() const noexcept;
        std::shared_ptr<DeviceContext> context() const noexcept;
    private:
        friend class ExprVkEvaluatePipeline;
        struct State;
        explicit Candidate(std::shared_ptr<State>);
        std::shared_ptr<State> state_;
    };

    // Runs one segment. program holds the packed image (see ExprVkPackProgram
    // in exprVkProgram.h); fields/owners hold the evaluation inputs. outSlots
    // is the output slot count (elements x registers/components); the output
    // buffer is host-visible so intermediate segments read back directly.
    std::unique_ptr<Candidate> Begin(std::shared_ptr<const ChargedBuffer> program,
                                     std::shared_ptr<const ChargedBuffer> fields,
                                     std::shared_ptr<const ChargedBuffer> owners,
                                     uint32_t count, uint32_t outSlots,
                                     VkResult* result = nullptr,
                                     BeforeSubmit beforeSubmit = {});

private:
    struct Native;
    explicit ExprVkEvaluatePipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_EXPR_VK_EVALUATE_H
