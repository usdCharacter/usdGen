// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_LENGTH_SCALE_PIPELINE_H
#define USDGEN_VULKAN_LENGTH_SCALE_PIPELINE_H

#include "chargedBuffer.h"
#include <cstdint>
#include <functional>
#include <vector>

namespace usdGen::vulkan {
// Queue-owner confined fixed-topology Length SCALE kernel.  The supplied
// module is a trusted, externally spirv-val-validated lengthScale.comp ABI.
class LengthScalePipeline final
    : public std::enable_shared_from_this<LengthScalePipeline> {
public:
    // Create() remains scale-only. CreateWithSet() installs a separately
    // trusted absolute-length module; it never reinterprets scale SPIR-V.
    class Candidate;
    using BeforeSubmit = std::function<bool()>;
    struct MinimumControls {
        enum class Method : uint32_t { Scale, CutExtend };
        enum class Rebuild : uint32_t { KeepParam, Reparam };
        float value = 1, minimum = 0;
        bool absolute = false;
        Method method = Method::Scale;
        Rebuild rebuild = Rebuild::KeepParam;
    };
    struct LiteralV1Controls {
        using Method = MinimumControls::Method;
        using Rebuild = MinimumControls::Rebuild;
        float value = 1, minimum = 0;
        bool absolute = false;
        Method method = Method::Scale;
        Rebuild rebuild = Rebuild::KeepParam;
        float randomLo = 1, randomHi = 1;
        int32_t seed = 0;
    };
    struct EnvelopeV1Controls {
        using Method = LiteralV1Controls::Method;
        using Rebuild = LiteralV1Controls::Rebuild;
        float value = 1, minimum = 0;
        bool absolute = false;
        Method method = Method::Scale;
        Rebuild rebuild = Rebuild::KeepParam;
        float randomLo = 1, randomHi = 1;
        int32_t seed = 0;
        float blend = 1, maskAmount = 1;
        bool cullOnly = false;
    };
    // Packed fields contain tightly packed float32/uint32 values. Random is
    // two float32 components; enabled is a canonical uint32 boolean.
    struct DeviceField {
        std::shared_ptr<const ChargedBuffer> data;
        uint32_t domain = 0; // 0 literal, 1 Groom, 2 Primitive, 4 Point.
    };
    struct DeviceV1Controls : EnvelopeV1Controls {
        float cullThreshold = 0;
        bool enabled = true;
        DeviceField valueField, randomField, minimumField, maskField,
            cullThresholdField, enabledField;
    };
    static std::shared_ptr<LengthScalePipeline> CreateWithDeviceV1(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& scaleCode,
        std::vector<uint32_t> const& setCode, std::vector<uint32_t> const& cutCode,
        std::vector<uint32_t> const& reparamCode, std::vector<uint32_t> const& minimumCode,
        std::vector<uint32_t> const& literalV1Code, std::vector<uint32_t> const& envelopeV1Code,
        std::vector<uint32_t> const& deviceV1Code, VkResult* = nullptr);
    bool HasDeviceV1() const noexcept;
    std::unique_ptr<Candidate> BeginDeviceV1(
        std::shared_ptr<const ChargedBuffer> points,
        std::shared_ptr<const ChargedBuffer> offsets,
        std::shared_ptr<const ChargedBuffer> hairT,
        std::shared_ptr<const ChargedBuffer> stableIds, uint32_t curveCount,
        uint32_t pointCount, DeviceV1Controls const&, VkResult* = nullptr,
        BeforeSubmit = {});
    static std::shared_ptr<LengthScalePipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const&,
        VkResult* = nullptr);
    static std::shared_ptr<LengthScalePipeline> CreateWithSet(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& scaleCode,
        std::vector<uint32_t> const& setCode,
        VkResult* = nullptr);
    // Installs independently trusted set and cut/extend modules.  The latter
    // shares the length kernel ABI but has its own native lifetime.
    static std::shared_ptr<LengthScalePipeline> CreateWithCutExtend(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& scaleCode,
        std::vector<uint32_t> const& setCode,
        std::vector<uint32_t> const& cutCode, VkResult* = nullptr);
    static std::shared_ptr<LengthScalePipeline> CreateWithReparam(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& scaleCode,
        std::vector<uint32_t> const& setCode,
        std::vector<uint32_t> const& cutCode,
        std::vector<uint32_t> const& reparamCode, VkResult* = nullptr);
    static std::shared_ptr<LengthScalePipeline> CreateWithMinimum(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& scaleCode,
        std::vector<uint32_t> const& setCode,
        std::vector<uint32_t> const& cutCode,
        std::vector<uint32_t> const& reparamCode,
        std::vector<uint32_t> const& minimumCode, VkResult* = nullptr);
    static std::shared_ptr<LengthScalePipeline> CreateWithLiteralV1(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& scaleCode,
        std::vector<uint32_t> const& setCode,
        std::vector<uint32_t> const& cutCode,
        std::vector<uint32_t> const& reparamCode,
        std::vector<uint32_t> const& minimumCode,
        std::vector<uint32_t> const& literalV1Code, VkResult* = nullptr);
    static std::shared_ptr<LengthScalePipeline> CreateWithEnvelopeV1(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& scaleCode,
        std::vector<uint32_t> const& setCode,
        std::vector<uint32_t> const& cutCode,
        std::vector<uint32_t> const& reparamCode,
        std::vector<uint32_t> const& minimumCode,
        std::vector<uint32_t> const& literalV1Code,
        std::vector<uint32_t> const& envelopeV1Code, VkResult* = nullptr);
    ~LengthScalePipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;
    std::unique_ptr<Candidate> Begin(
        std::shared_ptr<const ChargedBuffer> points,
        std::shared_ptr<const ChargedBuffer> offsets, uint32_t curveCount,
        uint32_t pointCount, float factor, VkResult* = nullptr,
        BeforeSubmit = {});
    bool HasSet() const noexcept;
    bool HasCutExtend() const noexcept;
    bool HasReparam() const noexcept;
    bool HasMinimum() const noexcept;
    bool HasLiteralV1() const noexcept;
    bool HasEnvelopeV1() const noexcept;
    std::unique_ptr<Candidate> BeginSet(
        std::shared_ptr<const ChargedBuffer> points,
        std::shared_ptr<const ChargedBuffer> offsets, uint32_t curveCount,
        uint32_t pointCount, float target, VkResult* = nullptr,
        BeforeSubmit = {});
    std::unique_ptr<Candidate> BeginCutExtend(
        std::shared_ptr<const ChargedBuffer> points,
        std::shared_ptr<const ChargedBuffer> offsets, uint32_t curveCount,
        uint32_t pointCount, float value, bool absolute, VkResult* = nullptr,
        BeforeSubmit = {});
    std::unique_ptr<Candidate> BeginReparam(
        std::shared_ptr<const ChargedBuffer> points,
        std::shared_ptr<const ChargedBuffer> offsets,
        std::shared_ptr<const ChargedBuffer> hairT, uint32_t curveCount,
        uint32_t pointCount, float value, bool absolute, VkResult* = nullptr,
        BeforeSubmit = {});
    std::unique_ptr<Candidate> BeginMinimum(
        std::shared_ptr<const ChargedBuffer> points,
        std::shared_ptr<const ChargedBuffer> offsets,
        std::shared_ptr<const ChargedBuffer> hairT, uint32_t curveCount,
        uint32_t pointCount, MinimumControls const&, VkResult* = nullptr,
        BeforeSubmit = {});
    std::unique_ptr<Candidate> BeginLiteralV1(
        std::shared_ptr<const ChargedBuffer> points,
        std::shared_ptr<const ChargedBuffer> offsets,
        std::shared_ptr<const ChargedBuffer> hairT,
        std::shared_ptr<const ChargedBuffer> stableIds, uint32_t curveCount,
        uint32_t pointCount, LiteralV1Controls const&, VkResult* = nullptr,
        BeforeSubmit = {});
    std::unique_ptr<Candidate> BeginEnvelopeV1(
        std::shared_ptr<const ChargedBuffer> points,
        std::shared_ptr<const ChargedBuffer> offsets,
        std::shared_ptr<const ChargedBuffer> hairT,
        std::shared_ptr<const ChargedBuffer> stableIds, uint32_t curveCount,
        uint32_t pointCount, EnvelopeV1Controls const&, VkResult* = nullptr,
        BeforeSubmit = {});
private:
    struct Native;
    explicit LengthScalePipeline(std::shared_ptr<Native>);
    static std::shared_ptr<LengthScalePipeline> CreateImpl(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const&, bool reparam,
        VkResult*, bool minimumFlavor = false, bool literalV1 = false,
        bool envelopeV1 = false, bool deviceV1 = false);
    std::unique_ptr<Candidate> BeginImpl(std::shared_ptr<Native>,
        std::shared_ptr<const ChargedBuffer>, std::shared_ptr<const ChargedBuffer>,
        uint32_t, uint32_t, float, VkResult*, BeforeSubmit,
        uint32_t computePhase = 1,
        std::shared_ptr<const ChargedBuffer> hairT = {},
        MinimumControls const* minimumControls = nullptr,
        std::shared_ptr<const ChargedBuffer> stableIds = {},
        LiteralV1Controls const* literalV1Controls = nullptr,
        EnvelopeV1Controls const* envelopeV1Controls = nullptr,
        DeviceV1Controls const* deviceV1Controls = nullptr);
    std::shared_ptr<Native> native_;
    std::shared_ptr<Native> setNative_;
    std::shared_ptr<Native> cutNative_;
    std::shared_ptr<Native> reparamNative_;
    std::shared_ptr<Native> minimumNative_;
    std::shared_ptr<Native> literalV1Native_;
    std::shared_ptr<Native> envelopeV1Native_;
    std::shared_ptr<Native> deviceV1Native_;
};

class LengthScalePipeline::Candidate final {
public:
    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    VkResult Poll(uint32_t* semanticStatus = nullptr);
    std::shared_ptr<const ChargedBuffer> output() const noexcept;
    std::shared_ptr<const ChargedBuffer> keep() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    std::shared_ptr<const ChargedBuffer> offsetsOwner() const noexcept;
    std::shared_ptr<const ChargedBuffer> hairTOwner() const noexcept;
    std::shared_ptr<const ChargedBuffer> stableIdsOwner() const noexcept;
    uint32_t count() const noexcept;
    uint32_t curveCount() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    bool succeeded() const noexcept;
    bool usesReparam() const noexcept;
    bool usesHairT() const noexcept;
    bool usesStableIds() const noexcept;
    void Quarantine() noexcept;
private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class LengthScalePipeline;
};
} // namespace usdGen::vulkan
#endif
