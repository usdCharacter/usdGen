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

// Status codes for RootFrameGather. Mirrors CudaRootFrameGather.
enum class RootFrameGatherSemantic : uint32_t {
    Ok = 0,
    MissingId = 1,
};

char const* RootFrameGatherSemanticName(RootFrameGatherSemantic) noexcept;

// RootFrameGatherPipeline re-sorts base per-root frames by survivor stable-ids.
// For each survivor id it binary-searches the (ascending) base id list and
// copies all six planes (origin/tangent/binormal/normal/valid/drop).
//
// Queue-owner confined. Inputs must be immutable and have completed upload on
// this queue before Begin. Neither Begin nor Poll waits for device completion
// beyond Poll's own synchronous status readback.
//
// Inputs must have:
// - sourceIds strictly ascending (guaranteed by upstream TopologyPipeline)
// - survivors strictly ascending
// - all float3 planes packed as float[] (12-byte stride, not std430 vec3[])
class RootFrameGatherPipeline final : public std::enable_shared_from_this<RootFrameGatherPipeline> {
public:
    class Candidate;

    // Owner-side admission checkpoint immediately before the first native
    // submit. Runs once only for a fully recorded candidate.
    using BeforeSubmit = std::function<bool()>;

    // Everything needed to begin a root-frame gather.
    struct BeginInfo {
        // Source (base) side — all size baseCount.
        std::shared_ptr<const ChargedBuffer> sourceIds;      // u64[baseCount] ascending
        std::shared_ptr<const ChargedBuffer> sourceOrigins;  // float[3*baseCount]
        std::shared_ptr<const ChargedBuffer> sourceTangent;  // float[3*baseCount]
        std::shared_ptr<const ChargedBuffer> sourceBinormal; // float[3*baseCount]
        std::shared_ptr<const ChargedBuffer> sourceNormal;   // float[3*baseCount]
        std::shared_ptr<const ChargedBuffer> sourceValid;    // u32[baseCount]
        std::shared_ptr<const ChargedBuffer> sourceDrop;     // u32[baseCount]
        uint32_t baseCount = 0;

        // Survivor side — all size survivorCount.
        std::shared_ptr<const ChargedBuffer> survivorIds;    // u64[survivorCount] ascending
        uint32_t survivorCount = 0;

        // All float3 planes are stored as scalar float[] (packed 12-byte stride).
    };

    // `spv` is the trusted, externally spirv-val-validated rootFrameGather.comp SPIR-V.
    static std::shared_ptr<RootFrameGatherPipeline> Create(
        std::shared_ptr<DeviceContext>, std::vector<uint32_t> const& spv, VkResult* result = nullptr);
    ~RootFrameGatherPipeline();
    std::shared_ptr<DeviceContext> const& context() const noexcept;

    // Validates, allocates output storage, records the gather dispatch,
    // and returns a candidate that owns all inputs and outputs until Poll.
    // On semantic failure (missing id), returns nullptr and sets *semantic.
    std::unique_ptr<Candidate> Begin(BeginInfo info, VkResult* result = nullptr,
        RootFrameGatherSemantic* semantic = nullptr, BeforeSubmit beforeSubmit = {});

private:
    struct Native;
    explicit RootFrameGatherPipeline(std::shared_ptr<Native>);
    std::shared_ptr<Native> native_;
};

// One candidate owns a single gather operation. It holds every input buffer
// so the whole dependency graph stays alive until Poll or destruction.
class RootFrameGatherPipeline::Candidate final {
public:
    // Result planes: 6 output buffers + status. All owned by the candidate.
    struct Output {
        std::shared_ptr<const ChargedBuffer> origins;   // float[3*survivorCount]
        std::shared_ptr<const ChargedBuffer> tangent;   // float[3*survivorCount]
        std::shared_ptr<const ChargedBuffer> binormal;  // float[3*survivorCount]
        std::shared_ptr<const ChargedBuffer> normal;    // float[3*survivorCount]
        std::shared_ptr<const ChargedBuffer> valid;     // u32[survivorCount]
        std::shared_ptr<const ChargedBuffer> drop;      // u32[survivorCount]
    };

    ~Candidate();
    Candidate(Candidate const&) = delete;
    Candidate& operator=(Candidate const&) = delete;
    Candidate(Candidate&&) = delete;
    Candidate& operator=(Candidate&&) = delete;

    // VK_SUCCESS means device proof obtained; `semantic` must be Ok before
    // output is exposed. VK_NOT_READY preserves the candidate.
    VkResult Poll(RootFrameGatherSemantic* semantic = nullptr);
    // Exposes the output planes only when proved and semantically clean.
    Output output() const noexcept;
    bool succeeded() const noexcept;
    std::shared_ptr<const ChargedBuffer> inputOwner() const noexcept;
    std::shared_ptr<DeviceContext> context() const noexcept;
    uint32_t survivorCount() const noexcept;
    void Quarantine() noexcept;

private:
    struct State;
    explicit Candidate(std::shared_ptr<State>);
    std::shared_ptr<State> state_;
    friend class RootFrameGatherPipeline;
};

} // namespace usdGen::vulkan
