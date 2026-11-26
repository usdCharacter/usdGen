// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// Descriptor compiler for the explicitly injected Vulkan geometry-value path.
#ifndef USDGEN_VULKAN_EXECUTION_PLAN_H
#define USDGEN_VULKAN_EXECUTION_PLAN_H

#include "usdGenCurves/vulkan/curveGrowPipeline.h"
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGenExpr/vulkan/exprVkProgram.h"
#include "usdGen/executionBackend.h"
#include "usdGenCurves/vulkan/vkStages.h"
#include <array>

#include <memory>
#include <vector>

namespace usdGen::vulkan {

// Literal scalar controls accepted by the flat Width shader. Ramps, maps,
// profiles and expression-driven controls remain excluded from this lane.
struct VulkanLiteralWidthControls {
    float width = 0.01f;
    bool replace = true;
    float mask = 1.0f;
    std::array<float, 257> profile = [] {
        std::array<float, 257> p{};
        p.fill(1.0f);
        return p;
    }();
    bool extended = false;
};
struct VulkanLiteralNoiseControls {
    float magnitude = 0.05f;
    float frequency = 3.0f;
    float correlation = 0.5f;
    int32_t octaves = 1;
    float lacunarity = 2.0f;
    float gain = 0.5f;
    float preserveLength = 1.0f;
    float mask = 1.0f;
    int32_t seed = 0;
    bool cumulative = false;
    bool enabled = true;
    std::array<float, 257> profile = [] {
        std::array<float, 257> p{};
        p.fill(1.0f);
        return p;
    }();
};
// RBF Deform is a topology-preserving points rewrite. Samples (rest + posed)
// are the host-side fixed-point fit (SelectSamples + CubicField::Bind/Solve)
// frozen at plan compile; the pipeline only applies displacement at runtime.
struct VulkanDeformControls {
    std::vector<float> restSamples;  // 3*sampleCount, desc space
    std::vector<float> posedSamples; // 3*sampleCount, desc space
    std::vector<float> rootTargets;  // 3*curveCount, desc space
    // Stable IDs address targets independently of source sorting/dropped roots.
    std::vector<uint64_t> rootStableIds;
    // Assigned by native job preparation; the immutable primitive plane stays
    // private and follows every downstream GPU topology gather.
    std::string rootTargetPlane;
    int sampleCount = 0; // [4,100] host-fit, [1,1024] device path
    float mask = 1.0f;
    float groomEnvelope = 1.0f;
    bool lockRoots = true;
    // Phase-2 RBF device path (rbfVk hook): true when the gap validator
    // (rbfVkPlan.h) posed source-local samples for the device solver.
    // Samples are source-local in both paths here (identity == desc).
    bool devicePath = false;
};

struct VulkanSourceWidthStage {
    enum class Kind { Width, LengthScale, WidthBlend, LengthCull, Noise, Deform, Resample, Grow };
    enum class LengthMode : uint32_t { Scale = 0, Set = 1 };
    enum class LengthMethod : uint32_t { Scale = 0, CutExtend = 1 };
    enum class LengthRebuild : uint32_t { KeepParam = 0, Reparam = 1 };
    Kind kind = Kind::Width;
    // Muted operators alias their input: no dispatch, no new value. The
    // flag is plan data so toggling never recompiles native state.
    bool disabled = false;
    // A dynamic source resample resolving to 0 shares its layout while still
    // publishing the topology value promised by the cached plan.
    bool publishTopologyRevision = false;
    bool enabledLiteral = true;
    bool emptyGenerator = false;
    // Value 0 is Source; stage i publishes value i+1. Binary inputs retain
    // their authored left/right order even when the stage list is sorted.
    uint32_t input = 0, rightInput = 0;
    UsdGenNodeKey key;
    VulkanLiteralWidthControls width;
    std::vector<ExprVkCompiledBinding> expressionBindings;
    expr::Context expressionControls;
    float factor = 1.0f, blend = 0.5f;
    LengthMode lengthMode = LengthMode::Scale;
    LengthMethod lengthMethod = LengthMethod::Scale;
    LengthRebuild lengthRebuild = LengthRebuild::KeepParam;
    float minRemainingLength = 0.0f;
    float randomLo = 1.0f, randomHi = 1.0f;
    int32_t randomSeed = 0;
    // EnvelopeV1 is deliberately literal-only. These are copied onto every
    // lowered stage so native admission never needs to revisit authoring.
    float lengthBlend = 1.0f, lengthMaskAmount = 1.0f;
    bool lengthCullOnly = false;
    bool lengthDeviceControls = false;
    bool lengthKeepInput = false;
    bool requiresNonWidthProof = false;
    float cullThreshold = 0.0f;
    VulkanLiteralNoiseControls noise;
    VulkanDeformControls deform;
    // Uniform output CVs per curve for Kind::Resample (>= 2). Matches the
    // CUDA CudaCurveResample target contract (0 = no stage, 1 invalid).
    uint32_t resampleTarget = 0;
    CurveGrowPipeline::Controls grow;
    std::string growTargetPlane;
};

// All source options admitted by this compiler.  These are copied out for a
// later executor instead of being treated as compile-time no-ops.
struct VulkanSourceControls {
    bool useRest = true;
    bool surfaceCage = false;
    uint64_t maxCurves = 0;
    uint64_t maxCvs = 0;
    bool hasAuthoredRest = false;
    bool restFromCurrentPoints = false;
    bool stableIdsSynthesized = false;
    TfToken idSource = UsdGenVkTokens().primvar;
    std::string expectEpoch;
    TfToken staleAction = UsdGenVkTokens().warn;
    TfToken rebind = UsdGenVkTokens().onError;
    // CurveSource resampleTo literal. 0 keeps the source layout (no stage);
    // >= 2 lowers to an implicit leading internal Resample stage.
    int resampleTo = 0;
};

// Typed, native-free payload behind UsdGenExecutionPlanHandle::Payload().
// It owns a full immutable descriptor capture as well as the selected C3
// source snapshot, so the native executor never consults a live graph. The
// historical class name is retained for compatibility with linear callers.
class VulkanSourceWidthPlan final {
  public:
    std::shared_ptr<const UsdGenGraphSpec> const &Descriptor() const noexcept {
        return descriptor_;
    }
    UsdGenCurveSetInputView const &Source() const noexcept { return source_; }
    UsdGenNodeKey SourceNodeKey() const noexcept { return sourceNodeKey_; }
    UsdGenNodeKey WidthNodeKey() const noexcept { return widthNodeKey_; }
    VulkanSourceControls const &SourceControls() const noexcept { return sourceControls_; }
    VulkanLiteralWidthControls const &Width() const noexcept { return width_; }
    bool HasLength() const noexcept { return lengthNodeKey_.IsValid(); }
    float LengthFactor() const noexcept { return lengthFactor_; }
    UsdGenNodeKey LengthNodeKey() const noexcept { return lengthNodeKey_; }
    std::vector<VulkanSourceWidthStage> const &Steps() const noexcept { return steps_; }
    size_t IntermediateCount() const noexcept { return steps_.empty() ? 0 : steps_.size() - 1; }
    uint64_t CurveGeneration() const noexcept { return curveGeneration_; }
    uint64_t TopologyVersion() const noexcept { return topologyVersion_; }
    uint64_t ValueVersion() const noexcept { return valueVersion_; }
    // Root capture produces this bundle only for a non-empty source with a
    // surface relationship or authored root channels.  The legacy
    // surface-free/no-root shape deliberately preserves its prior metadata.
    // It is deliberately separate from named channels: rootPrim/rootUV and
    // the private T/B/N frame are executor-owned structural data.
    bool HasRootBindings() const noexcept { return hasRootBindings_; }

  private:
    friend std::shared_ptr<const UsdGenExecutionPlanHandle>
    CompileVulkanSourceWidthPlan(UsdGenGraphSpec const &, UsdGenDiagnostics *);
    VulkanSourceWidthPlan(std::shared_ptr<const UsdGenGraphSpec>, UsdGenCurveSetInputView, UsdGenNodeKey,
                          UsdGenNodeKey, VulkanSourceControls, VulkanLiteralWidthControls, uint64_t,
                          uint64_t, uint64_t, bool) noexcept;
    std::shared_ptr<const UsdGenGraphSpec> descriptor_;
    UsdGenCurveSetInputView source_;   // holds its record's columns; descriptor_ holds the rest
    UsdGenNodeKey sourceNodeKey_, widthNodeKey_;
    VulkanSourceControls sourceControls_;
    VulkanLiteralWidthControls width_;
    UsdGenNodeKey lengthNodeKey_;
    float lengthFactor_ = 1.0f;
    std::vector<VulkanSourceWidthStage> steps_;
    uint64_t curveGeneration_ = 0, topologyVersion_ = 0, valueVersion_ = 0;
    bool hasRootBindings_ = false;
};

// Descriptor-only compilation: no backend registration or native allocation.
// Supports one CurveSource and bounded Width/LengthScale/WidthBlend stages,
// topologically sorted from arbitrary authored order while preserving ordered
// binary inputs. Every authored node must reach the selected terminal.
// Joins across distinct Length lineages require native non-width equality
// proof; other operator families are not admitted.
std::shared_ptr<const UsdGenExecutionPlanHandle>
CompileVulkanSourceWidthPlan(UsdGenGraphSpec const &, UsdGenDiagnostics * = nullptr);

} // namespace usdGen::vulkan

#endif // USDGEN_VULKAN_EXECUTION_PLAN_H
