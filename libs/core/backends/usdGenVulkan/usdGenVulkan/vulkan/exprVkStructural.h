// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_VULKAN_EXPR_VK_STRUCTURAL_H
#define USDGEN_VULKAN_EXPR_VK_STRUCTURAL_H
#include "completionService.h"
#include "exprVkRun.h"
#include <map>
namespace usdGen::vulkan {
// Small groom-domain structural packets (sample counts/source mode), read only
// after GPU proof. Geometry and point/primitive fields never cross this
// boundary.
using ExprVkStructuralValues = std::map<TfToken, uint32_t>;
void EvaluateExprVkStructural(std::shared_ptr<ExprVkContextPipeline>,
                              std::shared_ptr<ExprVkEvaluatePipeline>,
                              std::shared_ptr<VulkanCompletionService>,
                              std::vector<ExprVkCompiledBinding>, expr::Context,
                              std::function<void(ExprVkStructuralValues, VkResult)>);
} // namespace usdGen::vulkan
#endif
