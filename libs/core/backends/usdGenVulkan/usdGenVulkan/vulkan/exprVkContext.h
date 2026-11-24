// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// usdGen — Vulkan expression context pipeline (exprVkContext.comp).
//
// Builds one domain-specialized field set from curve geometry, mirroring
// gpu::CudaExpressionContext::Build: host-side shape admission (InvalidGeometry
// vs InvalidChannel, same conditions), then a validate + owners/arc phase and
// a field-materialization phase. Field layout (which variables exist, the
// $value literal region, host-step injection regions) is computed by the
// caller with ExprVkLayoutFields; this only runs the device phases. The $value
// literal and later injections are uploaded by the evaluator into the
// host-visible field buffer after this proves idle, so no upload ever races a
// dispatch. Groom evaluations run no dispatches (CUDA validates nothing for
// groom either) and prove immediately.
#ifndef USDGEN_VULKAN_EXPR_VK_CONTEXT_H
#define USDGEN_VULKAN_EXPR_VK_CONTEXT_H

#include "chargedBuffer.h"
#include "exprVkProgram.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {

enum class ExprVkContextStatus { Ok, InvalidArgument, InvalidGeometry, InvalidChannel, DeviceError };

// Device curve geometry. points holds pointCount float3; stableIds holds
// curveCount uint64 (read by the shader as lo/hi word pairs). Null/absent
// optionals materialize no field, exactly as on CUDA.
struct ExprVkCurveGeometry {
    std::shared_ptr<const ChargedBuffer> points;
    std::shared_ptr<const ChargedBuffer> restPoints;
    std::shared_ptr<const ChargedBuffer> widths;
    std::shared_ptr<const ChargedBuffer> curveOffsets;
    std::shared_ptr<const ChargedBuffer> stableIds;
    size_t curveCount = 0;
    size_t pointCount = 0;
};

struct ExprVkGeometryChannels {
    std::shared_ptr<const ChargedBuffer> hairT;
    std::shared_ptr<const ChargedBuffer> rootUV;
    std::shared_ptr<const ChargedBuffer> rootN;
    std::shared_ptr<const ChargedBuffer> rootT;
    std::shared_ptr<const ChargedBuffer> rootB;
    std::shared_ptr<const ChargedBuffer> rootPrim;
};

class ExprVkContextPipeline {
public:
    using BeforeSubmit = std::function<bool()>;
    static std::shared_ptr<ExprVkContextPipeline> Create(
        std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code,
        VkResult* result = nullptr);
    ~ExprVkContextPipeline();
    ExprVkContextPipeline(ExprVkContextPipeline const&) = delete;
    ExprVkContextPipeline& operator=(ExprVkContextPipeline const&) = delete;
    std::shared_ptr<DeviceContext> const& context() const noexcept;

    class Candidate {
    public:
        ~Candidate();
        Candidate(Candidate const&) = delete;
        Candidate& operator=(Candidate const&) = delete;
        void Quarantine() noexcept;
        bool succeeded() const noexcept;
        // Nonblocking proof; writes the semantic status (0 ok, 1 geometry,
        // 2 channel) once proved.
        VkResult Poll(uint32_t* semanticStatus);
        // Blocking proof for synchronous orchestration (mirrors the CUDA
        // Finish fence wait); a timeout leaves the candidate pending.
        VkResult Wait(uint64_t timeoutNs, uint32_t* semanticStatus);
        std::shared_ptr<ChargedBuffer> fields() const noexcept;
        std::shared_ptr<const ChargedBuffer> owners() const noexcept;
        std::shared_ptr<DeviceContext> context() const noexcept;
    private:
        friend class ExprVkContextPipeline;
        struct State;
        explicit Candidate(std::shared_ptr<State>);
        std::shared_ptr<State> state_;
    };

    // Builds the field set described by layout for domain (n = 1/curves/
    // points). count must equal the layout's n. Always sets status; returns
    // a candidate only for Ok (device phases or immediate groom proof).
    std::unique_ptr<Candidate> Begin(ExprVkCurveGeometry const& geometry,
                                     ExprVkGeometryChannels const& channels,
                                     expr::Domain domain, uint32_t count,
                                     ExprVkFieldLayout const& layout,
                                     ExprVkContextStatus* status,
                                     VkResult* result = nullptr,
                                     BeforeSubmit beforeSubmit = {});

private:
    struct Native;
    explicit ExprVkContextPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_EXPR_VK_CONTEXT_H
