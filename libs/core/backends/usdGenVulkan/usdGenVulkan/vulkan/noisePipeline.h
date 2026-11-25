// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Copyright 2026 Autodesk, Inc. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "usdGen/vulkan/chargedBuffer.h"

#include <functional>
#include <memory>
#include <vector>

namespace usdGen::vulkan {

// Status codes for the noise operator. Mirrors CUDA noise.cu:
// 0 = clean, 1 = kBadOffsets, 2 = kNonFinite, 3 = kBadValue.
enum class NoiseSemantic : uint32_t {
    Ok = 0,
    BadOffsets = 1,
    NonFinite = 2,
    BadValue = 3,
};

char const* NoiseSemanticName(NoiseSemantic) noexcept;

// Queue-owner confined. Inputs must be immutable and have completed upload on
// this queue before Begin. Neither Begin nor Poll waits for device completion
// beyond Begin's own synchronous validation-readback step.
//
// The generate dispatch uses FP64; the device context must have
// shaderFloat64Enabled (factory contract).
class NoisePipeline final : public std::enable_shared_from_this<NoisePipeline> {
public:
    class Candidate;

    // Owner-side admission checkpoint immediately before the first native
    // submit. Runs once only for a fully recorded candidate; invalid input,
    // allocation/recording failures, and semantic rejection never invoke it.
    using BeforeSubmit = std::function<bool()>;

    // A scalar (float) field parameter. Either a groom literal or a device
    // data plane in one of the three domains.
    struct ScalarField {
        float literal = 0.0f;
        uint32_t domain = 1;  // 1=Groom, 2=Primitive, 4=Point
        std::shared_ptr<const ChargedBuffer> data;
        uint32_t count = 0;
    };

    // An integer field parameter (octaves, seed).
    struct IntField {
        int32_t literal = 0;
        uint32_t domain = 1;
        std::shared_ptr<const ChargedBuffer> data;
        uint32_t count = 0;
    };

    // A boolean field parameter (enabled, cumulative). Groom or Primitive
    // only; Point is rejected host-side.
    struct BoolField {
        uint32_t literal = 0;
        uint32_t domain = 1;
        std::shared_ptr<const ChargedBuffer> data;
        uint32_t count = 0;
    };

    struct BeginInfo {
        // Geometry planes (float3 packed as scalar float[]).
        std::shared_ptr<const ChargedBuffer> points;       // float[3*pointCount]
        std::shared_ptr<const ChargedBuffer> restPoints;   // float[3*pointCount]
        std::shared_ptr<const ChargedBuffer> curveOffsets; // uint[curveCount+1]
        std::shared_ptr<const ChargedBuffer> stableIds;    // uint[2*curveCount]
        uint32_t curveCount = 0;
        uint32_t pointCount = 0;

        // Optional hairT (float[pointCount]); null means no hairT.
        std::shared_ptr<const ChargedBuffer> hairT;

        // Root frames. frameCount is derived from frameTangent when non-null.
        std::shared_ptr<const ChargedBuffer> frameTangent;   // float[3*frameCount]
        std::shared_ptr<const ChargedBuffer> frameBinormal;  // float[3*frameCount]
        std::shared_ptr<const ChargedBuffer> frameNormal;    // float[3*frameCount]
        std::shared_ptr<const ChargedBuffer> frameStableIds; // uint[2*frameCount] or null
        uint32_t frameCount = 0;

        // Magnitude profile (always 257 floats).
        std::shared_ptr<const ChargedBuffer> magnitudeProfile;

        // Field parameters.
        ScalarField magnitude, frequency, correlation, lacunarity, gain,
                    preserveLength, mask;
        IntField octaves, seed;
        BoolField enabled, cumulative;
    };

    // Both SPIR-V modules must be trusted, externally spirv-val-validated
    // noiseValidate.comp (validate) and noiseGenerate.comp (generate) with
    // the documented binding/push-constant ABIs. The generate module uses
    // FP64; the context must have shaderFloat64Enabled.
    static std::shared_ptr<NoisePipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& validateSpirv,
        std::vector<uint32_t> const& generateSpirv, VkResult* result = nullptr);
    ~NoisePipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;

    // Performs host-side validation, records a synchronous validate dispatch
    // (readback status), and on success records the generate dispatch
    // asynchronously. Returns a candidate owning all inputs and the output
    // plane; on semantic failure returns nullptr and sets `semantic`.
    std::unique_ptr<Candidate> Begin(BeginInfo info, VkResult* result = nullptr,
        NoiseSemantic* semantic = nullptr, BeforeSubmit beforeSubmit = {});

private:
    struct Native;
    explicit NoisePipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

// One candidate owns a single noise operation. It holds every input buffer
// so the whole dependency graph stays alive until Poll or destruction.
class NoisePipeline::Candidate final {
public:
    // Result plane: the generated points.
    struct Output {
        std::shared_ptr<const ChargedBuffer> points;  // float[3*pointCount]
    };

    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;

    // VK_SUCCESS means device proof obtained; `semantic` must be Ok before
    // output is exposed. VK_NOT_READY preserves the candidate.
    VkResult Poll(NoiseSemantic* semantic = nullptr);
    // Exposes the output plane only when proved and semantically clean.
    Output output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    uint32_t pointCount() const noexcept;
    void Quarantine() noexcept;

private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class NoisePipeline;
};

} // namespace usdGen::vulkan
