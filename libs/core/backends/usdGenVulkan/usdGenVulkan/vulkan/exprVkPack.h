// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_VULKAN_EXPR_VK_PACK_H
#define USDGEN_VULKAN_EXPR_VK_PACK_H
#include "chargedBuffer.h"
#include <functional>
#include <vector>

namespace usdGen::vulkan {
// Converts evaluated 8-byte expression slots into native 4-byte fields without
// a host transfer. The low word preserves float, int and boolean bits exactly.
class ExprVkPackPipeline {
public:
    using BeforeSubmit = std::function<bool()>;
    static std::shared_ptr<ExprVkPackPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const&, VkResult* = nullptr);
    ~ExprVkPackPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    class Candidate {
    public:
        ~Candidate();
        VkResult Poll(uint32_t* semanticStatus = nullptr);
        bool succeeded() const noexcept;
        std::shared_ptr<const ChargedBuffer> output() const noexcept;
        void Quarantine() noexcept;
    private:
        friend class ExprVkPackPipeline;
        struct State;
        explicit Candidate(std::shared_ptr<State> state) : state_(std::move(state)) {}
        std::shared_ptr<State> state_;
    };
    std::unique_ptr<Candidate> Begin(std::shared_ptr<const ChargedBuffer>,
        uint32_t count, VkResult* = nullptr, BeforeSubmit = {});
private:
    struct Native;
    explicit ExprVkPackPipeline(std::shared_ptr<Native> native) : native_(std::move(native)) {}
    std::shared_ptr<Native> native_;
};
} // namespace usdGen::vulkan
#endif
