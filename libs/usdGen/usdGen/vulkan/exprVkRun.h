// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_VULKAN_EXPR_VK_RUN_H
#define USDGEN_VULKAN_EXPR_VK_RUN_H
#include "exprVkParameters.h"
#include <functional>
#include <string_view>
namespace usdGen::vulkan {
// Queue-owner state machine. Each Advance submits at most one dispatch;
// Poll only consumes a proved dispatch. No native completion wait occurs here.
class ExprVkRun {
  public:
    ExprVkRun(std::shared_ptr<ExprVkContextPipeline>, std::shared_ptr<ExprVkEvaluatePipeline>,
              std::vector<ExprVkCompiledBinding>, ExprVkCurveGeometry, ExprVkGeometryChannels,
              expr::Context, bool buildOwners = true);
    ~ExprVkRun();
    bool Advance(std::function<bool()> beforeSubmit, VkResult *);
    VkResult Poll();
    bool done() const noexcept;
    bool succeeded() const noexcept;
    bool pending() const noexcept;
    bool NeedsHostStep() const noexcept;
    VkResult ComputeHostStep();
    VkResult CommitHostStep();
    void Quarantine() noexcept;
    ExprVkParameterField const *Find(TfToken const &) const;
    /// By destination text, with or without "usdGen:". Never interns.
    ExprVkParameterField const *Find(std::string_view) const;
    std::shared_ptr<const ChargedBuffer> owners() const noexcept;

  private:
    struct State;
    std::unique_ptr<State> state_;
};
} // namespace usdGen::vulkan
#endif
