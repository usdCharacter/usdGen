// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/sourceWidthJob.h"
#include "usdGenCurves/vulkan/lengthEnvelope.h"
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "usdGen/registries/token.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <new>
#include <set>
#include <stdexcept>
#include <vector>
namespace usdGen::vulkan {
struct VulkanSourceWidthJob::Control final {
    UsdGenExecutionPipeline::CommandMailbox mailbox;
    UsdGenExecutionPipeline::CommandTicket terminalTicket;
    UsdGenExecutionPipeline::CommandTicket prepareTicket;
};
struct VulkanSourceWidthJob::PrepareRelay : std::enable_shared_from_this<PrepareRelay> {
    UsdGenExecutionPipeline *owner = nullptr;
    UsdGenExecutionPipeline::CommandTicket ticket;
    std::shared_ptr<VulkanSourceWidthJob> jobLifetime;
    std::shared_ptr<Relay> terminal;
    std::shared_ptr<const VulkanPreparedSource> prepared;
    std::atomic<bool> taskError{false}, failed{false};
    std::unique_ptr<std::shared_ptr<PrepareRelay>> quarantine;
    void Retain() noexcept {
        if (quarantine) {
            *quarantine = shared_from_this();
            (void)quarantine.release();
        }
    }
    void Fail() noexcept;
};
struct VulkanSourceWidthJob::Relay : std::enable_shared_from_this<Relay> {
    std::atomic<bool> settled{false}, failurePosted{false};
    Completion completion;
    std::shared_ptr<const void> requestLifetime;
    std::shared_ptr<VulkanSourceWidthJob> jobLifetime;
    std::unique_ptr<std::shared_ptr<Relay>> quarantine;
    void Settle(std::shared_ptr<const VulkanSourceGeneration> result, State state, VkResult status,
                bool retain) noexcept {
        if (settled.exchange(true, std::memory_order_acq_rel))
            return;
        auto self = shared_from_this();
        // Released here (tickets included) unless a reader still holds it.
        jobLifetime->control_.Reset();
        if (retain) {
            *quarantine = self;
            (void)quarantine.release(); // Preallocated retention: no native reads.
        }
        jobLifetime->state_.store(state, std::memory_order_release);
        if (retain) {
            // Preserve callback captures and request ownership as well as the
            // native job: none may last-drop on this exceptional caller thread.
            try {
                completion(std::move(result), state, status);
            } catch (...) {
            }
        } else {
            auto callback = std::move(completion);
            auto lifetime = std::move(requestLifetime);
            try {
                callback(std::move(result), state, status);
            } catch (...) {
            }
            jobLifetime.reset(); // Native cleanup remains owner-confined.
        }
    }
    void Fallback() noexcept { Settle({}, State::LostProof, VK_ERROR_DEVICE_LOST, true); }
};
void VulkanSourceWidthJob::PrepareRelay::Fail() noexcept {
    if (failed.exchange(true, std::memory_order_acq_rel))
        return;
    Retain();
    if (terminal)
        terminal->Fallback();
}
VulkanSourceWidthJob::VulkanSourceWidthJob(CreateInfo info) : info_(std::move(info)) {}
std::shared_ptr<VulkanSourceWidthJob> VulkanSourceWidthJob::Create(CreateInfo info,
                                                                   std::string *why) {
    if (!info.source.context || !info.widthPipeline || !info.requestLifetime ||
        info.widthPipeline->context() != info.source.context ||
        (info.lengthPipeline && info.lengthPipeline->context() != info.source.context) ||
        (info.lengthCompactionPipeline &&
         info.lengthCompactionPipeline->context() != info.source.context) ||
        (info.resampleVkPipeline && info.resampleVkPipeline->context() != info.source.context) ||
        (info.tilesPipeline && info.tilesPipeline->context() != info.source.context) ||
        (info.boundsPipeline && info.boundsPipeline->context() != info.source.context) ||
        (info.nonWidthComparePipeline &&
         info.nonWidthComparePipeline->context() != info.source.context) ||
        info.valueVersion <= info.source.source.valueVersion || !std::isfinite(info.width) ||
        info.width < 0 || info.replace > 1 ||
        (info.stages.empty() && info.lengthPipeline &&
         (!std::isfinite(info.lengthFactor) || info.lengthFactor < 0 ||
          info.lengthValueVersion <= info.source.source.valueVersion ||
          info.valueVersion <= info.lengthValueVersion))) {
        if (why)
            *why = "invalid source-width job controls or revision";
        return {};
    }
    // Identity serves the executor lane (capture order is authoritative and
    // uniqueness-validated); Sorted passes through for sorted captures. The
    // lookup order stays rejected until survivor lookup planes are plumbed.
    bool const tileOrderServable = info.tileOrder == PicktileVkTileOrder::IdentityCaptureOrder ||
                                   info.tileOrder == PicktileVkTileOrder::SortedSurvivorSubset;
    if (info.tilesPipeline && info.boundsPipeline && !tileOrderServable) {
        if (why)
            *why = "unsupported source-width job tile order";
        return {};
    }
    try {
        if (info.stages.empty()) {
            if (info.lengthPipeline) {
                VulkanSourceWidthStage length;
                length.kind = VulkanSourceWidthStage::Kind::LengthScale;
                length.factor = info.lengthFactor;
                info.stages.push_back(length);
                info.stageValueVersions.push_back(info.lengthValueVersion);
            }
            VulkanSourceWidthStage width;
            width.input = uint32_t(info.stages.size());
            width.width.width = info.width;
            width.width.replace = info.replace != 0;
            info.stages.push_back(width);
            info.stageValueVersions.push_back(info.valueVersion);
        }
        if (info.stages.size() > size_t(UINT32_MAX) - 2 ||
            info.stageValueVersions.size() != info.stages.size() ||
            info.stageValueVersions.back() != info.valueVersion)
            return {};
        auto previous = info.source.source.valueVersion;
        for (size_t i = 0; i < info.stages.size(); ++i) {
            auto &stage = info.stages[i];
            if (stage.publishTopologyRevision &&
                (!stage.disabled || stage.kind != VulkanSourceWidthStage::Kind::Resample))
                return {};
            bool const expressionsNeeded =
                !stage.expressionBindings.empty() ||
                (stage.kind == VulkanSourceWidthStage::Kind::Width && stage.width.extended);
            if (expressionsNeeded &&
                (!info.expressionContextPipeline || !info.expressionEvaluatePipeline ||
                 info.expressionContextPipeline->context() != info.source.context ||
                 info.expressionEvaluatePipeline->context() != info.source.context))
                return {};
            if (stage.kind == VulkanSourceWidthStage::Kind::Width && expressionsNeeded &&
                (!info.expressionWidthPipeline ||
                 info.expressionWidthPipeline->context() != info.source.context))
                return {};
            if (!stage.expressionBindings.empty() &&
                stage.kind != VulkanSourceWidthStage::Kind::Width &&
                (!info.expressionPackPipeline ||
                 info.expressionPackPipeline->context() != info.source.context))
                return {};
            if (stage.requiresNonWidthProof && !info.nonWidthComparePipeline)
                return {};
            if (stage.input > i || info.stageValueVersions[i] <= previous)
                return {};
            previous = info.stageValueVersions[i];
            if (stage.disabled) {
                // A muted stage aliases its input without dispatch, so no
                // control or capability admission applies. Only Width and
                // Length stages mute; blends and unknown kinds never do.
                // Structural ordering above still applies.
                if (stage.kind != VulkanSourceWidthStage::Kind::Width &&
                    stage.kind != VulkanSourceWidthStage::Kind::LengthScale &&
                    stage.kind != VulkanSourceWidthStage::Kind::LengthCull &&
                    stage.kind != VulkanSourceWidthStage::Kind::Noise &&
                    stage.kind != VulkanSourceWidthStage::Kind::Resample &&
                    stage.kind != VulkanSourceWidthStage::Kind::Grow &&
                    stage.kind != VulkanSourceWidthStage::Kind::Deform)
                    return {};
                continue;
            }
            // These values may originate from a typed plan, but Create is
            // also a direct native admission boundary. Never let a cast or
            // future enum extension select an unintended pipeline path.
            bool const invalidLengthMethod =
                (stage.lengthMethod != VulkanSourceWidthStage::LengthMethod::Scale &&
                 stage.lengthMethod != VulkanSourceWidthStage::LengthMethod::CutExtend);
            bool const invalidLengthMode =
                (stage.lengthMode != VulkanSourceWidthStage::LengthMode::Scale &&
                 stage.lengthMode != VulkanSourceWidthStage::LengthMode::Set);
            bool const invalidLengthRebuild =
                stage.lengthRebuild != VulkanSourceWidthStage::LengthRebuild::KeepParam &&
                stage.lengthRebuild != VulkanSourceWidthStage::LengthRebuild::Reparam;
            bool const invalidMinimum =
                !std::isfinite(stage.minRemainingLength) || stage.minRemainingLength < 0;
            bool const invalidRandom = !std::isfinite(stage.randomLo) ||
                                       !std::isfinite(stage.randomHi) || stage.randomLo < 0 ||
                                       stage.randomHi < 0;
            bool const invalidEnvelope = !std::isfinite(stage.lengthBlend) ||
                                         stage.lengthBlend < 0 || stage.lengthBlend > 1 ||
                                         !std::isfinite(stage.lengthMaskAmount) ||
                                         stage.lengthMaskAmount < 0 || stage.lengthMaskAmount > 1;
            if ((stage.kind == VulkanSourceWidthStage::Kind::LengthScale ||
                 stage.kind == VulkanSourceWidthStage::Kind::LengthCull) &&
                (invalidLengthMethod || invalidLengthMode || invalidLengthRebuild ||
                 invalidMinimum || invalidRandom || invalidEnvelope))
                return {};
            if (stage.kind == VulkanSourceWidthStage::Kind::Width) {
                if (!std::isfinite(stage.width.width) || stage.width.width < 0)
                    return {};
            } else if (stage.kind == VulkanSourceWidthStage::Kind::LengthScale) {
                if (!info.lengthPipeline)
                    return {};
                if (stage.lengthDeviceControls) {
                    if (!info.lengthPipeline->HasDeviceV1())
                        return {};
                } else {
                    bool const literalV1 = stage.randomLo != 1.0f || stage.randomHi != 1.0f;
                    bool const nonneutralEnvelope =
                        stage.lengthBlend != 1.0f || stage.lengthMaskAmount != 1.0f;
                    bool const hasEnvelope = info.lengthPipeline->HasEnvelopeV1();
                    bool const missingEnvelope = nonneutralEnvelope && !hasEnvelope;
                    bool const missingLiteral =
                        !nonneutralEnvelope && literalV1 && !info.lengthPipeline->HasLiteralV1();
                    bool const missingMinimum = !nonneutralEnvelope && !literalV1 &&
                                                stage.minRemainingLength > 0 &&
                                                !info.lengthPipeline->HasMinimum();
                    bool const missingReparam =
                        !nonneutralEnvelope && !literalV1 && stage.minRemainingLength == 0 &&
                        stage.lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend &&
                        stage.lengthRebuild == VulkanSourceWidthStage::LengthRebuild::Reparam &&
                        !info.lengthPipeline->HasReparam();
                    bool const missingCut =
                        !nonneutralEnvelope && !literalV1 && stage.minRemainingLength == 0 &&
                        stage.lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend &&
                        stage.lengthRebuild != VulkanSourceWidthStage::LengthRebuild::Reparam &&
                        !info.lengthPipeline->HasCutExtend();
                    bool const missingSet =
                        !nonneutralEnvelope && !literalV1 && stage.minRemainingLength == 0 &&
                        stage.lengthMethod != VulkanSourceWidthStage::LengthMethod::CutExtend &&
                        stage.lengthMode == VulkanSourceWidthStage::LengthMode::Set &&
                        !info.lengthPipeline->HasSet();
                    if (!info.lengthPipeline || !std::isfinite(stage.factor) || stage.factor < 0 ||
                        (stage.lengthCullOnly && !nonneutralEnvelope) || missingEnvelope ||
                        missingLiteral || missingMinimum || missingReparam || missingCut ||
                        missingSet)
                        return {};
                }
            } else if (stage.kind == VulkanSourceWidthStage::Kind::LengthCull) {
                if (stage.lengthKeepInput &&
                    (!info.lengthCompactionPipeline || !info.lengthCompactionPipeline->HasKeep()))
                    return {};
                bool const nonneutralEnvelope =
                    stage.lengthBlend != 1.0f || stage.lengthMaskAmount != 1.0f;
                bool const hasEnvelope =
                    info.lengthPipeline && info.lengthPipeline->HasEnvelopeV1();
                bool malformedEnvelopeLineage = false;
                if (nonneutralEnvelope) {
                    if (stage.input == 0)
                        malformedEnvelopeLineage = true;
                    else {
                        auto const &producer = info.stages[stage.input - 1];
                        bool const producerEnvelope =
                            producer.lengthBlend != 1.0f || producer.lengthMaskAmount != 1.0f;
                        malformedEnvelopeLineage =
                            producer.kind != VulkanSourceWidthStage::Kind::LengthScale ||
                            !producerEnvelope || producer.lengthBlend != stage.lengthBlend ||
                            producer.lengthMaskAmount != stage.lengthMaskAmount;
                    }
                }
                if (!info.lengthCompactionPipeline || !std::isfinite(stage.factor) ||
                    stage.factor < 0 || !std::isfinite(stage.cullThreshold) ||
                    stage.cullThreshold < 0 || !std::isfinite(stage.randomLo) ||
                    !std::isfinite(stage.randomHi) || stage.randomLo < 0 || stage.randomHi < 0 ||
                    stage.lengthCullOnly || malformedEnvelopeLineage ||
                    (nonneutralEnvelope && !hasEnvelope))
                    return {};
            } else if (stage.kind == VulkanSourceWidthStage::Kind::WidthBlend) {
                if (!info.widthBlendPipeline ||
                    info.widthBlendPipeline->context() != info.source.context ||
                    stage.rightInput > i || !std::isfinite(stage.blend) || stage.blend < 0 ||
                    stage.blend > 1)
                    return {};
            } else if (stage.kind == VulkanSourceWidthStage::Kind::Noise) {
                if (!info.noisePipeline || info.noisePipeline->context() != info.source.context)
                    return {};
                if (!std::isfinite(stage.noise.magnitude) || stage.noise.magnitude < 0 ||
                    !std::isfinite(stage.noise.frequency) || stage.noise.frequency <= 0 ||
                    !std::isfinite(stage.noise.correlation) || stage.noise.correlation < 0 ||
                    stage.noise.correlation > 1 || stage.noise.octaves < 1 ||
                    stage.noise.octaves > 6 || !std::isfinite(stage.noise.lacunarity) ||
                    stage.noise.lacunarity <= 1 || !std::isfinite(stage.noise.gain) ||
                    stage.noise.gain < 0 || stage.noise.gain > 1 ||
                    !std::isfinite(stage.noise.preserveLength) || stage.noise.preserveLength < 0 ||
                    stage.noise.preserveLength > 1 || !std::isfinite(stage.noise.mask) ||
                    stage.noise.mask < 0 || stage.noise.mask > 1)
                    return {};
            } else if (stage.kind == VulkanSourceWidthStage::Kind::Deform) {
                // Phase-2 RBF device path (rbfVk hook): gap admission.
                if (stage.deform.devicePath || info.rbfVkDeformPipeline) {
                    if (!info.rbfVkDeformPipeline ||
                        info.rbfVkDeformPipeline->context() != info.source.context)
                        return {};
                    if (stage.deform.sampleCount < 1 || stage.deform.sampleCount > 46336)
                        return {};
                } else {
                    if (!info.deformPipeline ||
                        info.deformPipeline->context() != info.source.context)
                        return {};
                    if (stage.deform.sampleCount < 4 || stage.deform.sampleCount > 100)
                        return {};
                }
                if (stage.deform.restSamples.size() != size_t(stage.deform.sampleCount) * 3 ||
                    stage.deform.posedSamples.size() != size_t(stage.deform.sampleCount) * 3)
                    return {};
                if (!std::isfinite(stage.deform.mask) || stage.deform.mask < 0 ||
                    stage.deform.mask > 1 || !std::isfinite(stage.deform.groomEnvelope) ||
                    stage.deform.groomEnvelope < 0 || stage.deform.groomEnvelope > 1)
                    return {};
                auto const &controls = stage.deform;
                size_t const sourceCurves = info.source.source.totalCurves;
                if (info.source.source.curveId.size() != sourceCurves)
                    return {};
                std::map<uint64_t, size_t> targetById;
                if (!controls.rootStableIds.empty()) {
                    if (controls.rootTargets.size() != controls.rootStableIds.size() * 3)
                        return {};
                    for (size_t c = 0; c < controls.rootStableIds.size(); ++c)
                        if (!targetById.emplace(controls.rootStableIds[c], c).second)
                            return {};
                } else if (controls.rootTargets.size() != sourceCurves * 3)
                    return {};
                // Choose a job-private name that cannot shadow an authored
                // channel. Target alignment reads the immutable CPU source
                // capture once; later compaction is entirely GPU-resident.
                std::set<std::string> names;
                for (auto const &plane : info.source.source.extraCv)
                    names.insert(plane.name.GetString());
                for (auto const &plane : info.source.source.extraCurve)
                    names.insert(plane.name.GetString());
                for (auto const &plane : info.source.additionalNamed)
                    names.insert(plane.metadata.name.GetString());
                // The plan interned the name; only a collision with an
                // authored channel (or a hand-built stage) interns here.
                UsdGenToken name = stage.deform.rootTargetPlane;
                if (name.IsEmpty() || names.count(name.GetString())) {
                    std::string text = "__usdGenDeformRoots" + std::to_string(i);
                    while (names.count(text))
                        text += "_";
                    name = UsdGenInternToken(text);
                }
                VulkanSourceNamedChannel targets;
                targets.metadata = {name,
                                    UsdGenDeviceValueType::Float32x3,
                                    UsdGenDeviceDomain::Primitive,
                                    sourceCurves,
                                    3,
                                    12,
                                    true,
                                    UsdGenDeviceChannelSemantic::Generic};
                targets.privateStructural = true;
                targets.bytes.resize(sourceCurves * 12);
                for (size_t c = 0; c < sourceCurves; ++c) {
                    size_t authored = c;
                    if (!targetById.empty()) {
                        auto found = targetById.find(info.source.source.curveId[c]);
                        if (found == targetById.end())
                            return {};
                        authored = found->second;
                    }
                    for (size_t component = 0; component < 3; ++component)
                        if (!std::isfinite(controls.rootTargets[authored * 3 + component]))
                            return {};
                    std::memcpy(targets.bytes.data() + c * 12,
                                controls.rootTargets.data() + authored * 3, 12);
                }
                stage.deform.rootTargetPlane = name;
                info.source.additionalNamed.push_back(std::move(targets));
            } else if (stage.kind == VulkanSourceWidthStage::Kind::Grow) {
                if (stage.emptyGenerator) {
                    if (!info.lengthCompactionPipeline || !info.lengthCompactionPipeline->HasKeep())
                        return {};
                    continue;
                }
                if (!info.growVkPipeline || info.growVkPipeline->context() != info.source.context)
                    return {};
                auto targets = CurveGrowPipeline::BuildTargets(
                    std::vector<uint64_t>(info.source.source.curveId.begin(),
                                          info.source.source.curveId.end()),
                    stage.grow);
                if (targets.size() != info.source.source.totalCurves)
                    return {};
                if (stage.growTargetPlane.IsEmpty())
                    stage.growTargetPlane =
                        UsdGenInternToken("__vulkanGrowTargets" + std::to_string(i));
                VulkanSourceNamedChannel plane;
                plane.privateStructural = true;
                plane.metadata = {stage.growTargetPlane,
                                  UsdGenDeviceValueType::Float32,
                                  UsdGenDeviceDomain::Primitive,
                                  targets.size(),
                                  1,
                                  4,
                                  true,
                                  UsdGenDeviceChannelSemantic::Generic};
                plane.bytes.resize(targets.size() * sizeof(float));
                if (!targets.empty())
                    std::memcpy(plane.bytes.data(), targets.data(), plane.bytes.size());
                info.source.additionalNamed.push_back(std::move(plane));
            } else if (stage.kind == VulkanSourceWidthStage::Kind::Resample) {
                if (!info.resampleVkPipeline ||
                    info.resampleVkPipeline->context() != info.source.context ||
                    stage.resampleTarget < 2)
                    return {};
            } else
                return {};
        }
        auto job = std::shared_ptr<VulkanSourceWidthJob>(new VulkanSourceWidthJob(std::move(info)));
        job->values_.reserve(job->info_.stages.size() + 1);
        return job;
    } catch (std::bad_alloc const &) {
        if (why)
            *why = "source-width job allocation failed";
        return {};
    }
}
bool VulkanSourceWidthJob::Start(UsdGenExecutionPipeline &owner,
                                 UsdGenExecutionPipeline::Cancellation cancel, Completion done) {
    if (!done || started_.exchange(true, std::memory_order_acq_rel))
        return false;
    if (info_.completionService && (info_.source.context != info_.completionService->context() ||
                                    info_.completionService->queueOwner() != &owner))
        return false;
    try {
        auto terminal = owner.ReserveCommandTicket();
        if (!terminal)
            return false;
        auto mailbox = owner.ReserveCommandMailbox();
        if (!mailbox)
            return false;
        UsdGenExecutionPipeline::CommandTicket prepare;
        if (info_.preparer) {
            prepare = owner.ReserveCommandTicket();
            if (!prepare)
                return false;
        }
        auto control = std::make_shared<Control>();
        control->mailbox = std::move(mailbox);
        control->terminalTicket = std::move(terminal);
        control->prepareTicket = std::move(prepare);
        auto relay = std::make_shared<Relay>();
        relay->quarantine = std::make_unique<std::shared_ptr<Relay>>();
        relay->completion = std::move(done);
        relay->requestLifetime = std::move(info_.requestLifetime);
        relay->jobLifetime = shared_from_this();
        owner_ = &owner;
        cancellation_ = std::move(cancel);
        relay_ = std::move(relay);
        control_.Store(std::move(control));
    } catch (...) {
        return false;
    }
    NotifyCompletion(); // Once admitted, even dispatch failure settles once.
    return true;
}
bool VulkanSourceWidthJob::NotifyCompletion() {
    if (!relay_ || relay_->settled.load(std::memory_order_acquire))
        return false;
    if (state() == State::Preparing)
        return true; // CPU task owns the next owner handoff.
    auto control = control_.Load();
    if (!control)
        return false;
    bool posted = false;
    try {
        auto self = shared_from_this();
        if (!owner_->CommandMailboxBroken(control->mailbox))
            posted = owner_->PostLatestCommand(control->mailbox, [self] { self->Advance(); });
    } catch (...) {
    }
    if (!posted)
        RouteFailure();
    return posted;
}
void VulkanSourceWidthJob::NotifyFailure(VkResult status) noexcept { RouteFailure(status); }
void VulkanSourceWidthJob::RouteFailure(VkResult status) noexcept {
    auto relay = relay_;
    if (relay->settled.load(std::memory_order_acquire) ||
        relay->failurePosted.exchange(true, std::memory_order_acq_rel))
        return;
    bool posted = false;
    try {
        auto control = control_.Take();
        if (!control) {
            relay->Fallback();
            return;
        }
        auto self = shared_from_this();
        posted = owner_->PostCommand(
            std::move(control->terminalTicket), [self, status] { self->FailOnOwner(status); },
            [relay] { relay->Fallback(); });
    } catch (...) {
    }
    if (!posted)
        relay->Fallback();
}
bool VulkanSourceWidthJob::Suppressed() const noexcept {
    return suppressed_.load(std::memory_order_acquire) || cancellation_.Superseded();
}
void VulkanSourceWidthJob::Terminal(std::shared_ptr<const VulkanSourceGeneration> result,
                                    State state, VkResult status) noexcept {
    Terminal(std::move(result), state, status, state_.load(std::memory_order_acquire));
}
void VulkanSourceWidthJob::Terminal(std::shared_ptr<const VulkanSourceGeneration> result,
                                    State state, VkResult status, State phase) noexcept {
    failurePhase_.store(phase, std::memory_order_release);
    relay_->Settle(std::move(result), state, status, false);
}
void VulkanSourceWidthJob::FailOnOwner(VkResult status) {
    if (relay_->settled.load(std::memory_order_acquire))
        return;
    // Break the job<->preparation relay anchor on the owner before terminal
    // settlement. A late worker completion then sees no job to adopt, while
    // unavailable transport keeps its explicitly quarantined relay alive.
    if (prepareRelay_) {
        prepareRelay_->jobLifetime.reset();
        prepareRelay_.reset();
    }
    if (upload_)
        upload_->Quarantine();
    if (width_)
        width_->Quarantine();
    if (expressions_)
        expressions_->Quarantine();
    if (expressionWidth_)
        expressionWidth_->Quarantine();
    if (length_)
        length_->Quarantine();
    if (cull_)
        cull_->Quarantine();
    if (blend_)
        blend_->Quarantine();
    if (compare_)
        compare_->Quarantine();
    if (noise_)
        noise_->Quarantine();
    if (deform_)
        deform_->Quarantine();
    if (rbfVkDeform_)
        rbfVkDeform_->Quarantine();
    if (resample_)
        resample_->Quarantine();
    if (grow_)
        grow_->Quarantine();
    if (expressionPack_)
        expressionPack_->Quarantine();
    Terminal({}, State::LostProof, status);
}
void VulkanSourceWidthJob::BeginPreparation() {
    if (!owner_->IsExecutingOwner() || !info_.preparer || prepareRelay_)
        return;
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    auto control = control_.Load();
    if (!control || !control->prepareTicket) {
        Terminal({}, State::Failed, VK_ERROR_OUT_OF_HOST_MEMORY);
        return;
    }
    auto relay = std::make_shared<PrepareRelay>();
    relay->owner = owner_;
    relay->ticket = std::move(control->prepareTicket);
    relay->jobLifetime = shared_from_this();
    relay->terminal = relay_;
    relay->quarantine = std::make_unique<std::shared_ptr<PrepareRelay>>();
    auto created = State::Created;
    if (!state_.compare_exchange_strong(created, State::Preparing, std::memory_order_acq_rel))
        return;
    VulkanSourcePrepareInfo cpu{std::move(info_.source.source), std::move(info_.source.geometry),
                                std::move(info_.source.additionalNamed)};
    prepareRelay_ = relay;
    UsdGenExecutionTaskGraph::Job task;
    task.cancellation = cancellation_;
    task.tasks.push_back({{}, [cpu = std::move(cpu), relay](auto const &, auto finish) mutable {
                              auto prepared = VulkanPreparedSource::Prepare(std::move(cpu));
                              if (!prepared) {
                                  finish({}, std::make_exception_ptr(
                                                 std::runtime_error("source preparation failed")));
                                  return;
                              }
                              finish([relay, prepared] { relay->prepared = prepared; }, {});
                          }});
    task.completion = [relay](UsdGenExecutionPipeline::Publish publication,
                              std::exception_ptr error) mutable {
        relay->taskError.store(bool(error), std::memory_order_release);
        bool posted = false;
        try {
            auto ticket = std::move(relay->ticket);
            posted = relay->owner->PostCommand(
                std::move(ticket),
                [relay, publication = std::move(publication)]() mutable {
                    if (publication)
                        publication();
                    auto job = relay->jobLifetime;
                    if (job)
                        job->TakePrepared(relay);
                },
                [relay] { relay->Fail(); });
        } catch (...) {
        }
        if (!posted)
            relay->Fail();
    };
    if (!info_.preparer->Submit(std::move(task))) {
        prepareRelay_.reset();
        Terminal({}, State::Failed, VK_ERROR_OUT_OF_HOST_MEMORY);
    }
}
void VulkanSourceWidthJob::TakePrepared(std::shared_ptr<PrepareRelay> relay) noexcept {
    if (!owner_ || !owner_->IsExecutingOwner() || relay != prepareRelay_)
        return;
    auto prepared = relay->prepared;
    relay->jobLifetime.reset(); // owner-only: worker relay can no longer own native job state.
    prepareRelay_.reset();
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (!relay_ || relay_->settled.load(std::memory_order_acquire))
        return;
    if (relay->taskError.load(std::memory_order_acquire) || !prepared) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    auto preparing = State::Preparing;
    if (!state_.compare_exchange_strong(preparing, State::Created, std::memory_order_acq_rel))
        return;
    prepared_ = std::move(prepared);
    Advance();
}
void VulkanSourceWidthJob::Advance() {
    if (relay_->settled.load(std::memory_order_acquire))
        return;
    try {
        switch (state()) {
        case State::ExpressionHostPending:
            return;
        case State::Created: {
            if (Suppressed()) {
                Terminal({}, State::Superseded, VK_SUCCESS);
                return;
            }
            if (info_.preparer && !prepared_) {
                BeginPreparation();
                return;
            }
            if (info_.completionService) {
                auto self = shared_from_this();
                watch_ = info_.completionService->Reserve(
                    self,
                    [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                        if (proof == VK_SUCCESS) {
                            self->watchProofReady_ = true;
                            return self->NotifyCompletion()
                                       ? VulkanCompletionService::DeliveryResult::Posted
                                       : VulkanCompletionService::DeliveryResult::Stale;
                        }
                        self->NotifyFailure(proof);
                        return VulkanCompletionService::DeliveryResult::LostProof;
                    },
                    [self](VkResult proof) { self->NotifyFailure(proof); });
                if (!watch_) {
                    RouteFailure();
                    return;
                }
            }
            VkResult status = VK_ERROR_INITIALIZATION_FAILED;
            upload_ = prepared_ ? VulkanSourceUpload::Create(info_.source.context,
                                                             std::move(prepared_), &status)
                                : VulkanSourceUpload::Create(std::move(info_.source), &status);
            if (!upload_) {
                if (watch_) {
                    info_.completionService->CancelBeforeSubmit(*watch_);
                    watch_.reset();
                }
                Terminal({}, State::Failed, status);
                return;
            }
            if (watch_ && !watch_->MarkPhaseSubmitted()) {
                watch_.reset();
                RouteFailure();
                return;
            }
            auto submitted = upload_->Submit();
            if (submitted != SourceGenerationStatus::Submitted) {
                if (watch_)
                    watch_.reset();
                Terminal({},
                         submitted == SourceGenerationStatus::LostProof ? State::LostProof
                                                                        : State::Failed,
                         VK_ERROR_INITIALIZATION_FAILED);
                return;
            }
            if (watch_ && !info_.completionService->Arm(*watch_, uint64_t(State::UploadPending))) {
                watch_.reset();
                RouteFailure();
                return;
            }
            auto expected = State::Created;
            state_.compare_exchange_strong(expected, State::UploadPending,
                                           std::memory_order_acq_rel);
            return;
        }
        case State::UploadPending: {
            if (watch_ && !watchProofReady_)
                return;
            if (watch_) {
                if (!watch_->Retire()) {
                    watch_.reset();
                    Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
                    return;
                }
                watch_.reset();
                watchProofReady_ = false;
            }
            auto proof = upload_->Poll();
            if (proof == SourceGenerationStatus::NotReady) {
                // A consumed service proof has no future notification. Its
                // proxy fence must prove this phase fence too; fail safely
                // if that contract is violated instead of silently stalling.
                if (info_.completionService)
                    FailOnOwner();
                return;
            }
            if (proof != SourceGenerationStatus::Ready) {
                Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
                return;
            }
            if (Suppressed()) {
                Terminal({}, State::Superseded, VK_SUCCESS);
                return;
            }
            base_ = upload_->TakeReady();
            if (!base_) {
                Terminal({}, State::Failed, VK_ERROR_OUT_OF_HOST_MEMORY);
                return;
            }
            values_.push_back(base_);
            lengthKeepValues_.push_back({});
            BeginStage();
            return;
        }
        case State::LengthPending:
            FinishLength();
            return;
        case State::CullCountsPending:
            FinishCull(false);
            return;
        case State::CullScatterPending:
            FinishCull(true);
            return;
        case State::WidthPending:
            FinishWidth();
            return;
        case State::ExpressionPending:
            FinishExpressions();
            return;
        case State::ExpressionWidthPending:
            FinishExpressionWidth();
            return;
        case State::ExpressionPackPending:
            FinishExpressionPack();
            return;
        case State::BlendPending:
            FinishBlend();
            return;
        case State::ComparePending:
            FinishCompare();
            return;
        case State::NoisePending:
            FinishNoise();
            return;
        case State::DeformPending:
            FinishDeform();
            return;
        case State::ResamplePending:
            FinishResample();
            return;
        case State::GrowPending:
            FinishGrow();
            return;
        case State::TileFinalizePending:
            FinishTiles();
            return;
        default:
            return;
        }
    } catch (...) {
        FailOnOwner();
    }
}
void VulkanSourceWidthJob::BeginStage() {
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    auto const &stage = info_.stages[stageIndex_];
    base_ = values_[stage.input];
    nextLengthKeep_.reset();
    compare_.reset();
    if (stage.disabled) {
        // Muted operators share input planes. FinishStage publishes the
        // authoritative revision without submitting native work.
        FinishStage(values_[stage.input]);
        return;
    }
    if (stageIndex_ == 0)
        stageFrom_ = State::UploadPending;
    else {
        auto previous = info_.stages[stageIndex_ - 1].kind;
        stageFrom_ = previous == VulkanSourceWidthStage::Kind::Width         ? State::WidthPending
                     : previous == VulkanSourceWidthStage::Kind::LengthScale ? State::LengthPending
                     : previous == VulkanSourceWidthStage::Kind::LengthCull
                         ? State::CullScatterPending
                     : previous == VulkanSourceWidthStage::Kind::Noise    ? State::NoisePending
                     : previous == VulkanSourceWidthStage::Kind::Resample ? State::ResamplePending
                     : previous == VulkanSourceWidthStage::Kind::Deform   ? State::DeformPending
                                                                          : State::BlendPending;
    }
    expressions_.reset();
    expressionWidth_.reset();
    expressionProfile_.reset();
    packedFields_.clear();
    packedIndex_ = 0;
    expressionPack_.reset();
    if (!stage.expressionBindings.empty() ||
        (stage.kind == VulkanSourceWidthStage::Kind::Width && stage.width.extended)) {
        BeginExpressions();
        return;
    }
    DispatchStage();
}
void VulkanSourceWidthJob::DispatchStage() {
    auto const &stage = info_.stages[stageIndex_];
    if (stage.kind == VulkanSourceWidthStage::Kind::Width && (stage.width.extended || expressions_))
        BeginExpressionWidth();
    else if (stage.kind == VulkanSourceWidthStage::Kind::Width)
        BeginWidth();
    else if (stage.kind == VulkanSourceWidthStage::Kind::LengthScale)
        BeginLength();
    else if (stage.kind == VulkanSourceWidthStage::Kind::LengthCull)
        BeginCull();
    else if (stage.kind == VulkanSourceWidthStage::Kind::Noise)
        BeginNoise();
    else if (stage.kind == VulkanSourceWidthStage::Kind::Deform)
        BeginDeform();
    else if (stage.kind == VulkanSourceWidthStage::Kind::Resample)
        BeginResample();
    else if (stage.kind == VulkanSourceWidthStage::Kind::Grow && stage.emptyGenerator)
        BeginCull();
    else if (stage.kind == VulkanSourceWidthStage::Kind::Grow)
        BeginGrow();
    else
        BeginBlend();
}
void VulkanSourceWidthJob::FinishStage(std::shared_ptr<const VulkanSourceGeneration> value) {
    if (value && info_.stages[stageIndex_].disabled &&
        info_.stages[stageIndex_].kind == VulkanSourceWidthStage::Kind::Deform) {
        value =
            VulkanSourceGeneration::WithDeformAlias(value, info_.stageValueVersions[stageIndex_]);
        if (!value) {
            Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
            return;
        }
    }
    if (value && info_.stages[stageIndex_].publishTopologyRevision) {
        value = VulkanSourceGeneration::WithTopologyVersion(value,
                                                            info_.stageValueVersions[stageIndex_]);
        if (!value) {
            Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
            return;
        }
    }
    if (value && value->valueVersion() != info_.stageValueVersions[stageIndex_]) {
        value =
            VulkanSourceGeneration::WithValueVersion(value, info_.stageValueVersions[stageIndex_]);
        if (!value) {
            Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
            return;
        }
    }
    values_.push_back(value);
    lengthKeepValues_.push_back(std::move(nextLengthKeep_));
    ++stageIndex_;
    if (stageIndex_ == info_.stages.size()) {
        if (NeedsTiles(*value)) {
            BeginTiles();
            return;
        }
        Terminal(std::move(value), State::Ready, VK_SUCCESS);
        return;
    }
    BeginStage();
}
void VulkanSourceWidthJob::BeginCompare() {
    if (!info_.nonWidthComparePipeline) {
        Terminal({}, State::Failed, VK_ERROR_FEATURE_NOT_PRESENT);
        return;
    }
    VkResult status = VK_ERROR_INITIALIZATION_FAILED;
    if (info_.completionService) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    bool marked = false;
    NonWidthComparePipeline::BeforeSubmit before;
    if (watch_)
        before = [this, &marked] {
            marked = watch_ && watch_->MarkPhaseSubmitted();
            return marked;
        };
    compare_ = info_.nonWidthComparePipeline->Begin(
        base_, values_[info_.stages[stageIndex_].rightInput], &status, std::move(before));
    if (!compare_) {
        if (watch_) {
            if (!marked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, marked ? State::LostProof : State::Failed, status);
        return;
    }
    if (watch_ &&
        (!marked || !info_.completionService->Arm(*watch_, uint64_t(State::ComparePending)))) {
        watch_.reset();
        RouteFailure();
        return;
    }
    auto expected = stageFrom_;
    state_.compare_exchange_strong(expected, State::ComparePending, std::memory_order_acq_rel);
}
void VulkanSourceWidthJob::FinishCompare() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    auto proof = compare_->Poll();
    if (proof == VK_NOT_READY) {
        if (info_.completionService)
            FailOnOwner();
        return;
    }
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (!compare_->succeeded()) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    stageFrom_ = State::ComparePending;
    BeginBlend();
}
void VulkanSourceWidthJob::BeginBlend() {
    auto const &stage = info_.stages[stageIndex_];
    auto const &right = values_[stage.rightInput];
    if (base_->NonWidthIdentity() != right->NonWidthIdentity()) {
        if (!compare_) {
            BeginCompare();
            return;
        }
        if (!compare_->succeeded() || compare_->leftOwner() != base_ ||
            compare_->rightOwner() != right) {
            Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
            return;
        }
    }
    VkResult status = VK_ERROR_INITIALIZATION_FAILED;
    if (info_.completionService && base_->pointCount()) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    bool marked = false;
    WidthBlendPipeline::BeforeSubmit before;
    if (watch_)
        before = [this, &marked] {
            marked = watch_ && watch_->MarkPhaseSubmitted();
            return marked;
        };
    blend_ = info_.widthBlendPipeline->Begin(base_->PlaneOwner(UsdGenCurveChannels().width), right->PlaneOwner(UsdGenCurveChannels().width),
                                             base_->pointCount(), stage.blend, &status,
                                             std::move(before));
    if (!blend_) {
        if (watch_) {
            if (!marked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, marked ? State::LostProof : State::Failed, status);
        return;
    }
    if (watch_ &&
        (!marked || !info_.completionService->Arm(*watch_, uint64_t(State::BlendPending)))) {
        watch_.reset();
        RouteFailure();
        return;
    }
    auto expected = stageFrom_;
    state_.compare_exchange_strong(expected, State::BlendPending, std::memory_order_acq_rel);
    if (!base_->pointCount())
        FinishBlend();
}
void VulkanSourceWidthJob::FinishBlend() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    uint32_t semantic = 0;
    auto proof = blend_->Poll(&semantic);
    if (proof == VK_NOT_READY) {
        if (info_.completionService)
            FailOnOwner();
        return;
    }
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (semantic) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto value = VulkanSourceGeneration::WithWidthBlend(
        base_, values_[info_.stages[stageIndex_].rightInput], *blend_,
        info_.stageValueVersions[stageIndex_], nullptr, compare_.get());
    if (!value) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    FinishStage(std::move(value));
}
void VulkanSourceWidthJob::BeginCull(bool scatter) {
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    VkResult status = VK_ERROR_INITIALIZATION_FAILED;
    if (info_.completionService) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    bool phaseMarked = false;
    LengthCompactionPipeline::BeforeSubmit beforeSubmit;
    if (watch_)
        beforeSubmit = [this, &phaseMarked] {
            phaseMarked = watch_ && watch_->MarkPhaseSubmitted();
            return phaseMarked;
        };
    bool submitted = false;
    if (scatter)
        submitted = cull_->BeginScatter(&status, std::move(beforeSubmit));
    else {
        auto const &stage = info_.stages[stageIndex_];
        float const effectiveThreshold =
            ResolveLengthEnvelope(stage.lengthBlend, stage.lengthMaskAmount) == 0.0f
                ? 0.0f
                : stage.cullThreshold;
        std::shared_ptr<const ChargedBuffer> emptyKeep;
        if (stage.emptyGenerator && base_->curveCount()) {
            auto ctx = base_->context();
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = VkDeviceSize(base_->curveCount()) * 4;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            auto keep = ChargedBuffer::Create(
                ctx, bi, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch, &status);
            void *mapped = nullptr;
            if (!keep ||
                vkMapMemory(ctx->device(), keep->memory(), 0, bi.size, 0, &mapped) != VK_SUCCESS) {
                if (watch_) {
                    info_.completionService->CancelBeforeSubmit(*watch_);
                    watch_.reset();
                }
                Terminal({}, State::Failed, VK_ERROR_OUT_OF_HOST_MEMORY);
                return;
            }
            std::memset(mapped, 0, size_t(bi.size));
            vkUnmapMemory(ctx->device(), keep->memory());
            emptyKeep = std::move(keep);
        }
        cull_ = stage.emptyGenerator ? info_.lengthCompactionPipeline->BeginWithKeep(
                                           base_, emptyKeep, &status, std::move(beforeSubmit))
                : stage.lengthKeepInput
                    ? info_.lengthCompactionPipeline->BeginWithKeep(
                          base_, lengthKeepValues_[stage.input], &status, std::move(beforeSubmit))
                    : info_.lengthCompactionPipeline->Begin(base_, effectiveThreshold, &status,
                                                            std::move(beforeSubmit));
        submitted = bool(cull_);
    }
    if (!submitted) {
        // Both cull submissions cross the native-lifetime admission boundary.
        // A failed scatter return after that boundary cannot prove that no
        // reads were queued, so retain the candidate's full native graph.
        // This also makes the job robust if the pipeline reports a post-mark
        // submission failure before it can quarantine itself.
        if (phaseMarked && cull_)
            cull_->Quarantine();
        if (watch_) {
            if (!phaseMarked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, phaseMarked ? State::LostProof : State::Failed, status);
        return;
    }
    State const pending = scatter ? State::CullScatterPending : State::CullCountsPending;
    if (watch_ && (!phaseMarked || !info_.completionService->Arm(*watch_, uint64_t(pending)))) {
        watch_.reset();
        RouteFailure();
        return;
    }
    auto expected = scatter ? State::CullCountsPending : stageFrom_;
    state_.compare_exchange_strong(expected, pending, std::memory_order_acq_rel);
}
void VulkanSourceWidthJob::FinishCull(bool scatter) {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    uint32_t semantic = 0;
    auto proof = scatter ? cull_->PollScatter(&semantic) : cull_->PollCounts(&semantic);
    if (proof == VK_NOT_READY) {
        if (info_.completionService)
            FailOnOwner();
        return;
    }
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (semantic) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    if (!scatter) {
        BeginCull(true);
        return;
    }
    auto child = VulkanSourceGeneration::WithCompacted(
        base_, *cull_, info_.stageValueVersions[stageIndex_], nullptr,
        info_.stages[stageIndex_].emptyGenerator &&
            info_.stages[stageIndex_].kind == VulkanSourceWidthStage::Kind::Grow);
    if (!child) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    FinishStage(std::move(child));
}
void VulkanSourceWidthJob::BeginResample() {
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    VkResult status = VK_ERROR_INITIALIZATION_FAILED;
    if (info_.completionService) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    bool phaseMarked = false;
    ResampleVkPipeline::BeforeSubmit beforeSubmit;
    if (watch_)
        beforeSubmit = [this, &phaseMarked] {
            phaseMarked = watch_ && watch_->MarkPhaseSubmitted();
            return phaseMarked;
        };
    auto const &stage = info_.stages[stageIndex_];
    resample_ = info_.resampleVkPipeline->Begin(base_, stage.resampleTarget, &status,
                                                std::move(beforeSubmit));
    if (!resample_) {
        // A null candidate submitted nothing observable: a post-mark queue
        // failure already quarantines the candidate state internally.
        if (watch_) {
            if (!phaseMarked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, phaseMarked ? State::LostProof : State::Failed, status);
        return;
    }
    if (watch_ && (!phaseMarked ||
                   !info_.completionService->Arm(*watch_, uint64_t(State::ResamplePending)))) {
        watch_.reset();
        RouteFailure();
        return;
    }
    auto expected = stageFrom_;
    state_.compare_exchange_strong(expected, State::ResamplePending, std::memory_order_acq_rel);
}
void VulkanSourceWidthJob::FinishResample() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    uint32_t semantic = 0;
    auto proof = resample_->Poll(&semantic);
    if (proof == VK_NOT_READY) {
        if (info_.completionService)
            FailOnOwner();
        return;
    }
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (semantic) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto const &stage = info_.stages[stageIndex_];
    auto child = VulkanSourceGeneration::WithResampled(base_, *resample_, stage.resampleTarget,
                                                       info_.stageValueVersions[stageIndex_]);
    if (!child) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    FinishStage(std::move(child));
}
bool VulkanSourceWidthJob::NeedsTiles(VulkanSourceGeneration const &value) const noexcept {
    if (!info_.tilesPipeline || !info_.boundsPipeline)
        return false;
    if (value.curveCount() == 0 || value.pointCount() == 0)
        return false;
    // Cull-lane follow-up: compaction changes survivor cardinality, which
    // needs survivor planes + order selection before tiles can run. Until
    // that lands, only identity-cardinality lanes finalize tiles.
    if (values_.empty() || !values_.front() || value.curveCount() != values_.front()->curveCount())
        return false;
    for (auto const &tile : value.geometry().tiles)
        if (!tile.boundsValid)
            return true;
    return value.geometry().tiles.empty();
}
void VulkanSourceWidthJob::BeginTiles() {
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    auto const &value = values_.back();
    auto const &capture = values_.front();
    uint32_t const captureCurves = capture->curveCount();
    uint32_t const survivorCurves = value->curveCount();
    uint32_t const survivorPoints = value->pointCount();
    auto captureIds = capture->PlaneOwner(UsdGenCurveChannels().stableIds);
    auto survivorIds = value->PlaneOwner(UsdGenCurveChannels().stableIds);
    auto survivorOffsets = value->PlaneOwner(UsdGenCurveChannels().curveOffsets);
    PicktileVkTileRequirements requirements{};
    if (!captureIds || !survivorIds || !survivorOffsets ||
        !GetPicktileVkTileRequirements(info_.tileChunkSize, info_.tileTarget, captureCurves,
                                       survivorCurves, survivorPoints, &requirements) ||
        requirements.tileCount == 0) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    VkResult status = VK_ERROR_INITIALIZATION_FAILED;
    if (info_.completionService) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    bool phaseMarked = false;
    PicktileVkTilesPipeline::BeforeSubmit beforeSubmit;
    if (watch_)
        beforeSubmit = [this, &phaseMarked] {
            phaseMarked = watch_ && watch_->MarkPhaseSubmitted();
            return phaseMarked;
        };
    tiles_ = info_.tilesPipeline->Begin(std::move(captureIds), std::move(survivorIds),
                                        std::move(survivorOffsets), nullptr, nullptr, captureCurves,
                                        survivorCurves, survivorPoints, info_.tileOrder,
                                        requirements, &status, std::move(beforeSubmit));
    if (!tiles_) {
        if (watch_) {
            if (!phaseMarked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, phaseMarked ? State::LostProof : State::Failed, status);
        return;
    }
    if (watch_ && (!phaseMarked ||
                   !info_.completionService->Arm(*watch_, uint64_t(State::TileFinalizePending)))) {
        watch_.reset();
        RouteFailure();
        return;
    }
    tileCount_ = requirements.tileCount;
    auto expected = state_.load(std::memory_order_acquire);
    state_.compare_exchange_strong(expected, State::TileFinalizePending, std::memory_order_acq_rel);
}
void VulkanSourceWidthJob::BeginTileBounds() {
    auto const &value = values_.back();
    auto spans = tiles_->spans();
    auto points = value->PlaneOwner(UsdGenCurveChannels().points);
    auto widths = value->PlaneOwner(UsdGenCurveChannels().width);
    if (!spans || !points || !widths || tileCount_ == 0 || tileCount_ > size_t(INT32_MAX) ||
        spans->sizeBytes() != VkDeviceSize(tileCount_) * sizeof(PicktileVkTileSpan)) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    VkResult status = VK_ERROR_INITIALIZATION_FAILED;
    if (info_.completionService) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    bool phaseMarked = false;
    PicktileVkBoundsPipeline::BeforeSubmit beforeSubmit;
    if (watch_)
        beforeSubmit = [this, &phaseMarked] {
            phaseMarked = watch_ && watch_->MarkPhaseSubmitted();
            return phaseMarked;
        };
    // CUDA parity (Finalize): CatmullRom only when the capture topology says
    // so, otherwise BSpline. Linear is unrepresentable in the device basis.
    auto basis = UsdGenCurveLayoutBasisOf(value->geometry().layoutMetadata) == UsdGenCurveLayoutBasis::CatmullRom
                     ? PicktileVkTileBoundsBasis::CatmullRom
                     : PicktileVkTileBoundsBasis::BSpline;
    bounds_ = info_.boundsPipeline->Begin(std::move(points), std::move(widths), std::move(spans),
                                          value->pointCount(), uint32_t(tileCount_), basis, &status,
                                          std::move(beforeSubmit));
    if (!bounds_) {
        if (watch_) {
            if (!phaseMarked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, phaseMarked ? State::LostProof : State::Failed, status);
        return;
    }
    if (watch_ && (!phaseMarked ||
                   !info_.completionService->Arm(*watch_, uint64_t(State::TileFinalizePending)))) {
        watch_.reset();
        RouteFailure();
        return;
    }
}
void VulkanSourceWidthJob::FinishTiles() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    if (!bounds_) {
        uint32_t deviceStatus = UINT32_MAX;
        auto proof = tiles_->Poll(&deviceStatus);
        if (proof == VK_NOT_READY) {
            if (info_.completionService)
                FailOnOwner();
            return;
        }
        if (proof != VK_SUCCESS) {
            Terminal({}, State::LostProof, proof);
            return;
        }
        if (Suppressed()) {
            Terminal({}, State::Superseded, VK_SUCCESS);
            return;
        }
        // A rejected spans candidate leaves its buffer untouched by
        // contract; the job fails and publishes nothing (CUDA parity).
        if (deviceStatus != 0) {
            Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
            return;
        }
        BeginTileBounds();
        return;
    }
    uint32_t deviceStatus = UINT32_MAX;
    auto proof = bounds_->Poll(&deviceStatus);
    if (proof == VK_NOT_READY) {
        if (info_.completionService)
            FailOnOwner();
        return;
    }
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (deviceStatus != 0) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto spans = tiles_->spans();
    auto minimums = bounds_->minimums();
    auto maximums = bounds_->maximums();
    VkDeviceSize const spanBytes = VkDeviceSize(tileCount_) * sizeof(PicktileVkTileSpan);
    VkDeviceSize const extentBytes = VkDeviceSize(tileCount_) * 3 * sizeof(float);
    if (!spans || !minimums || !maximums || tileCount_ == 0 || spans->sizeBytes() != spanBytes ||
        minimums->sizeBytes() != extentBytes || maximums->sizeBytes() != extentBytes) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    // Owned picktile outputs are host-visible/coherent (tools path): map the
    // proved buffers directly instead of staging a copy.
    std::vector<PicktileVkTileSpan> hostSpans(tileCount_);
    std::vector<float> hostMins(tileCount_ * 3), hostMaxs(tileCount_ * 3);
    auto device = values_.back()->context()->device();
    void *mapped = nullptr;
    if (vkMapMemory(device, spans->memory(), 0, spanBytes, 0, &mapped) != VK_SUCCESS) {
        Terminal({}, State::Failed, VK_ERROR_MEMORY_MAP_FAILED);
        return;
    }
    std::memcpy(hostSpans.data(), mapped, spanBytes);
    vkUnmapMemory(device, spans->memory());
    if (vkMapMemory(device, minimums->memory(), 0, extentBytes, 0, &mapped) != VK_SUCCESS) {
        Terminal({}, State::Failed, VK_ERROR_MEMORY_MAP_FAILED);
        return;
    }
    std::memcpy(hostMins.data(), mapped, extentBytes);
    vkUnmapMemory(device, minimums->memory());
    if (vkMapMemory(device, maximums->memory(), 0, extentBytes, 0, &mapped) != VK_SUCCESS) {
        Terminal({}, State::Failed, VK_ERROR_MEMORY_MAP_FAILED);
        return;
    }
    std::memcpy(hostMaxs.data(), mapped, extentBytes);
    vkUnmapMemory(device, maximums->memory());
    std::vector<UsdGenDeviceTileMetadata> tiles;
    if (!PicktileVkPublishTileMetadata(hostSpans.data(), hostMins.data(), hostMaxs.data(),
                                       tileCount_, &tiles)) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto child = VulkanSourceGeneration::WithTiles(values_.back(), std::move(tiles));
    if (!child) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    Terminal(std::move(child), State::Ready, VK_SUCCESS);
}
void VulkanSourceWidthJob::BeginLength() {
    VkResult status = VK_ERROR_INITIALIZATION_FAILED;
    if (info_.completionService && base_->pointCount()) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    bool phaseMarked = false;
    LengthScalePipeline::BeforeSubmit beforeSubmit;
    if (watch_)
        beforeSubmit = [this, &phaseMarked] {
            phaseMarked = watch_ && watch_->MarkPhaseSubmitted();
            return phaseMarked;
        };
    auto const &stage = info_.stages[stageIndex_];
    bool const literalV1 = stage.randomLo != 1.0f || stage.randomHi != 1.0f;
    bool const nonneutralEnvelope = stage.lengthBlend != 1.0f || stage.lengthMaskAmount != 1.0f;
    if (stage.lengthDeviceControls) {
        LengthScalePipeline::DeviceV1Controls controls;
        controls.value = stage.factor;
        controls.minimum = stage.minRemainingLength;
        controls.absolute = stage.lengthMode == VulkanSourceWidthStage::LengthMode::Set;
        controls.method = stage.lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend
                              ? LengthScalePipeline::DeviceV1Controls::Method::CutExtend
                              : LengthScalePipeline::DeviceV1Controls::Method::Scale;
        controls.rebuild = stage.lengthRebuild == VulkanSourceWidthStage::LengthRebuild::Reparam
                               ? LengthScalePipeline::DeviceV1Controls::Rebuild::Reparam
                               : LengthScalePipeline::DeviceV1Controls::Rebuild::KeepParam;
        controls.randomLo = stage.randomLo;
        controls.randomHi = stage.randomHi;
        controls.seed = stage.randomSeed;
        controls.blend = stage.lengthBlend;
        controls.maskAmount = stage.lengthMaskAmount;
        controls.cullOnly = stage.lengthCullOnly;
        controls.cullThreshold = stage.cullThreshold;
        controls.enabled = stage.enabledLiteral;
        auto bind = [&](char const *name, auto &field) {
            if (auto f = PackedField(name)) {
                field.data = f->data;
                field.domain = uint32_t(f->domain);
            }
        };
        bind("length:value", controls.valueField);
        bind("length:random", controls.randomField);
        bind("minRemainingLength", controls.minimumField);
        bind("mask", controls.maskField);
        bind("cullThreshold", controls.cullThresholdField);
        bind("enabled", controls.enabledField);
        length_ = info_.lengthPipeline->BeginDeviceV1(
            base_->PlaneOwner(UsdGenCurveChannels().points), base_->PlaneOwner(UsdGenCurveChannels().curveOffsets),
            base_->PlaneOwner(UsdGenCurveChannels().hairT), base_->PlaneOwner(UsdGenCurveChannels().stableIds), base_->curveCount(),
            base_->pointCount(), std::move(controls), &status, std::move(beforeSubmit));
    } else
        length_ =
            nonneutralEnvelope
                ? info_.lengthPipeline->BeginEnvelopeV1(
                      base_->PlaneOwner(UsdGenCurveChannels().points), base_->PlaneOwner(UsdGenCurveChannels().curveOffsets),
                      base_->PlaneOwner(UsdGenCurveChannels().hairT), base_->PlaneOwner(UsdGenCurveChannels().stableIds),
                      base_->curveCount(), base_->pointCount(),
                      LengthScalePipeline::EnvelopeV1Controls{
                          stage.factor, stage.minRemainingLength,
                          stage.lengthMode == VulkanSourceWidthStage::LengthMode::Set,
                          stage.lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend
                              ? LengthScalePipeline::EnvelopeV1Controls::Method::CutExtend
                              : LengthScalePipeline::EnvelopeV1Controls::Method::Scale,
                          stage.lengthRebuild == VulkanSourceWidthStage::LengthRebuild::Reparam
                              ? LengthScalePipeline::EnvelopeV1Controls::Rebuild::Reparam
                              : LengthScalePipeline::EnvelopeV1Controls::Rebuild::KeepParam,
                          stage.randomLo, stage.randomHi, stage.randomSeed, stage.lengthBlend,
                          stage.lengthMaskAmount, stage.lengthCullOnly},
                      &status, std::move(beforeSubmit))
            : literalV1
                ? info_.lengthPipeline->BeginLiteralV1(
                      base_->PlaneOwner(UsdGenCurveChannels().points), base_->PlaneOwner(UsdGenCurveChannels().curveOffsets),
                      base_->PlaneOwner(UsdGenCurveChannels().hairT), base_->PlaneOwner(UsdGenCurveChannels().stableIds),
                      base_->curveCount(), base_->pointCount(),
                      LengthScalePipeline::LiteralV1Controls{
                          stage.factor, stage.minRemainingLength,
                          stage.lengthMode == VulkanSourceWidthStage::LengthMode::Set,
                          stage.lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend
                              ? LengthScalePipeline::LiteralV1Controls::Method::CutExtend
                              : LengthScalePipeline::LiteralV1Controls::Method::Scale,
                          stage.lengthRebuild == VulkanSourceWidthStage::LengthRebuild::Reparam
                              ? LengthScalePipeline::LiteralV1Controls::Rebuild::Reparam
                              : LengthScalePipeline::LiteralV1Controls::Rebuild::KeepParam,
                          stage.randomLo, stage.randomHi, stage.randomSeed},
                      &status, std::move(beforeSubmit))
            : stage.minRemainingLength > 0
                ? info_.lengthPipeline->BeginMinimum(
                      base_->PlaneOwner(UsdGenCurveChannels().points), base_->PlaneOwner(UsdGenCurveChannels().curveOffsets),
                      base_->PlaneOwner(UsdGenCurveChannels().hairT), base_->curveCount(), base_->pointCount(),
                      LengthScalePipeline::MinimumControls{
                          stage.factor, stage.minRemainingLength,
                          stage.lengthMode == VulkanSourceWidthStage::LengthMode::Set,
                          stage.lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend
                              ? LengthScalePipeline::MinimumControls::Method::CutExtend
                              : LengthScalePipeline::MinimumControls::Method::Scale,
                          stage.lengthRebuild == VulkanSourceWidthStage::LengthRebuild::Reparam
                              ? LengthScalePipeline::MinimumControls::Rebuild::Reparam
                              : LengthScalePipeline::MinimumControls::Rebuild::KeepParam},
                      &status, std::move(beforeSubmit))
            : stage.lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend &&
                    stage.lengthRebuild == VulkanSourceWidthStage::LengthRebuild::Reparam
                ? info_.lengthPipeline->BeginReparam(
                      base_->PlaneOwner(UsdGenCurveChannels().points), base_->PlaneOwner(UsdGenCurveChannels().curveOffsets),
                      base_->PlaneOwner(UsdGenCurveChannels().hairT), base_->curveCount(), base_->pointCount(),
                      stage.factor, stage.lengthMode == VulkanSourceWidthStage::LengthMode::Set,
                      &status, std::move(beforeSubmit))
            : stage.lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend
                ? info_.lengthPipeline->BeginCutExtend(
                      base_->PlaneOwner(UsdGenCurveChannels().points), base_->PlaneOwner(UsdGenCurveChannels().curveOffsets),
                      base_->curveCount(), base_->pointCount(), stage.factor,
                      stage.lengthMode == VulkanSourceWidthStage::LengthMode::Set, &status,
                      std::move(beforeSubmit))
            : stage.lengthMode == VulkanSourceWidthStage::LengthMode::Set
                ? info_.lengthPipeline->BeginSet(base_->PlaneOwner(UsdGenCurveChannels().points),
                                                 base_->PlaneOwner(UsdGenCurveChannels().curveOffsets),
                                                 base_->curveCount(), base_->pointCount(),
                                                 stage.factor, &status, std::move(beforeSubmit))
                : info_.lengthPipeline->Begin(base_->PlaneOwner(UsdGenCurveChannels().points),
                                              base_->PlaneOwner(UsdGenCurveChannels().curveOffsets),
                                              base_->curveCount(), base_->pointCount(),
                                              stage.factor, &status, std::move(beforeSubmit));
    if (!length_) {
        if (watch_) {
            if (!phaseMarked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, phaseMarked ? State::LostProof : State::Failed, status);
        return;
    }
    if (watch_ &&
        (!phaseMarked || !info_.completionService->Arm(*watch_, uint64_t(State::LengthPending)))) {
        watch_.reset();
        RouteFailure();
        return;
    }
    auto expected = stageFrom_;
    state_.compare_exchange_strong(expected, State::LengthPending, std::memory_order_acq_rel);
    if (!base_->pointCount())
        FinishLength();
}
void VulkanSourceWidthJob::FinishLength() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    uint32_t semantic = 0;
    auto proof = length_->Poll(&semantic);
    if (proof == VK_NOT_READY) {
        if (info_.completionService)
            FailOnOwner();
        return;
    }
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (semantic) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    if (info_.stages[stageIndex_].lengthDeviceControls)
        nextLengthKeep_ = length_->keep();
    auto child =
        VulkanSourceGeneration::WithPoints(base_, *length_, info_.stageValueVersions[stageIndex_]);
    if (!child) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    FinishStage(std::move(child));
}
void VulkanSourceWidthJob::BeginNoise() {
    VkResult status = VK_ERROR_INITIALIZATION_FAILED;
    if (info_.completionService && base_->pointCount()) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    bool phaseMarked = false;
    NoisePipeline::BeforeSubmit beforeSubmit;
    if (watch_)
        beforeSubmit = [this, &phaseMarked] {
            phaseMarked = watch_ && watch_->MarkPhaseSubmitted();
            return phaseMarked;
        };
    auto const &stage = info_.stages[stageIndex_];
    // The literal lane has no magnitude knots: a flat 257-entry profile of 1.0.
    float profile[257];
    for (int i = 0; i < 257; ++i)
        profile[i] = stage.noise.profile[size_t(i)];
    auto const &ctx = base_->context();
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = 257u * sizeof(float);
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    auto magnitudeProfile = ChargedBuffer::Create(
        ctx, bi, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch);
    if (!magnitudeProfile) {
        status = VK_ERROR_OUT_OF_HOST_MEMORY;
        if (watch_) {
            if (!phaseMarked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, phaseMarked ? State::LostProof : State::Failed, status);
        return;
    }
    {
        void *p = nullptr;
        if (vkMapMemory(ctx->device(), magnitudeProfile->memory(), 0, bi.size, 0, &p) !=
            VK_SUCCESS) {
            status = VK_ERROR_OUT_OF_HOST_MEMORY;
            if (watch_) {
                if (!phaseMarked)
                    info_.completionService->CancelBeforeSubmit(*watch_);
                watch_.reset();
            }
            Terminal({}, phaseMarked ? State::LostProof : State::Failed, status);
            return;
        }
        std::memcpy(p, profile, bi.size);
        vkUnmapMemory(ctx->device(), magnitudeProfile->memory());
    }
    NoisePipeline::BeginInfo info;
    info.points = base_->PlaneOwner(UsdGenCurveChannels().points);
    info.restPoints = base_->PlaneOwner(UsdGenCurveChannels().rest);
    info.curveOffsets = base_->PlaneOwner(UsdGenCurveChannels().curveOffsets);
    info.stableIds = base_->PlaneOwner(UsdGenCurveChannels().stableIds);
    info.curveCount = base_->curveCount();
    info.pointCount = base_->pointCount();
    info.hairT = base_->PlaneOwner(UsdGenCurveChannels().hairT);
    info.frameTangent = base_->SourceFrameOwner(UsdGenCurveChannels().sourceRootT);
    info.frameBinormal = base_->SourceFrameOwner(UsdGenCurveChannels().sourceRootB);
    info.frameNormal = base_->SourceFrameOwner(UsdGenCurveChannels().sourceRootN);
    info.frameStableIds = nullptr;
    info.frameCount = base_->curveCount();
    info.magnitudeProfile = std::move(magnitudeProfile);
    info.magnitude = {stage.noise.magnitude, 1, nullptr, 0};
    info.frequency = {stage.noise.frequency, 1, nullptr, 0};
    info.correlation = {stage.noise.correlation, 1, nullptr, 0};
    info.lacunarity = {stage.noise.lacunarity, 1, nullptr, 0};
    info.gain = {stage.noise.gain, 1, nullptr, 0};
    info.preserveLength = {stage.noise.preserveLength, 1, nullptr, 0};
    info.mask = {stage.noise.mask, 1, nullptr, 0};
    info.octaves = {stage.noise.octaves, 1, nullptr, 0};
    info.seed = {stage.noise.seed, 1, nullptr, 0};
    info.enabled = {stage.noise.enabled ? 1u : 0u, 1, nullptr, 0};
    info.cumulative = {stage.noise.cumulative ? 1u : 0u, 1, nullptr, 0};
    auto scalar = [&](char const *name, auto &field) {
        if (auto f = PackedField(name)) {
            field.data = f->data;
            field.count = uint32_t(f->count);
            field.domain = uint32_t(f->domain);
        }
    };
    scalar("noise:magnitude", info.magnitude);
    scalar("noise:frequency", info.frequency);
    scalar("noise:correlation", info.correlation);
    scalar("noise:lacunarity", info.lacunarity);
    scalar("noise:gain", info.gain);
    scalar("preserveLength", info.preserveLength);
    scalar("mask", info.mask);
    scalar("noise:octaves", info.octaves);
    scalar("noise:seed", info.seed);
    scalar("enabled", info.enabled);
    scalar("cumulative", info.cumulative);
    NoiseSemantic beginSemantic = NoiseSemantic::Ok;
    noise_ = info_.noisePipeline->Begin(std::move(info), &status, &beginSemantic,
                                        std::move(beforeSubmit));
    if (!noise_) {
        if (watch_) {
            if (!phaseMarked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, phaseMarked ? State::LostProof : State::Failed, status);
        return;
    }
    if (watch_ &&
        (!phaseMarked || !info_.completionService->Arm(*watch_, uint64_t(State::NoisePending)))) {
        watch_.reset();
        RouteFailure();
        return;
    }
    auto expected = stageFrom_;
    state_.compare_exchange_strong(expected, State::NoisePending, std::memory_order_acq_rel);
    if (!base_->pointCount())
        FinishNoise();
}
void VulkanSourceWidthJob::FinishNoise() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    NoiseSemantic semantic = NoiseSemantic::Ok;
    auto proof = noise_->Poll(&semantic);
    if (proof == VK_NOT_READY) {
        if (info_.completionService)
            FailOnOwner();
        return;
    }
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (!noise_->succeeded() || semantic != NoiseSemantic::Ok) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto child =
        VulkanSourceGeneration::WithNoise(base_, *noise_, info_.stageValueVersions[stageIndex_]);
    if (!child) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    FinishStage(std::move(child));
}
void VulkanSourceWidthJob::BeginDeform() {
    // Phase-2 RBF device path (rbfVk hook): gap configs run the device solver.
    if (info_.stages[stageIndex_].deform.devicePath || info_.rbfVkDeformPipeline) {
        BeginRbfVkDeform();
        return;
    }
    // Launch submits the evaluate proof without waiting; AdvanceDeform
    // watches its timeline value and submits the apply once it is proven.
    VkResult status = VK_ERROR_INITIALIZATION_FAILED;
    DeformPipeline::BeforeSubmit beforeSubmit;
    if (info_.completionService && base_->pointCount())
        beforeSubmit = [this] {
            deformMarked_ = watch_ && watch_->MarkPhaseSubmitted();
            return deformMarked_;
        };
    auto const &stage = info_.stages[stageIndex_];
    DeformPipeline::BeginInfo info;
    info.points = base_->PlaneOwner(UsdGenCurveChannels().points);
    info.curveOffsets = base_->curveCount() ? base_->PlaneOwner(UsdGenCurveChannels().curveOffsets) : nullptr;
    info.rootTargets = base_->SourceFrameOwner(stage.deform.rootTargetPlane);
    info.curveCount = base_->curveCount();
    info.pointCount = base_->pointCount();
    info.restSamples = stage.deform.restSamples;
    info.posedSamples = stage.deform.posedSamples;
    info.sampleCount = stage.deform.sampleCount;
    info.smoothing = 0.0;
    info.mask = {stage.deform.mask, 1, nullptr, 0};
    info.enabled = {stage.enabledLiteral ? 1u : 0u, 1, nullptr, 0};
    info.lockRoots = {stage.deform.lockRoots ? 1u : 0u, 1, nullptr, 0};
    info.groomEnvelope = stage.deform.groomEnvelope;
    auto bind = [&](char const *name, auto &field) {
        if (auto f = PackedField(name)) {
            field.data = f->data;
            field.count = uint32_t(f->count);
            field.domain = uint32_t(f->domain);
        }
    };
    bind("mask", info.mask);
    bind("enabled", info.enabled);
    bind("lockRoots", info.lockRoots);
    DeformSemantic beginSemantic = DeformSemantic::Ok;
    deformMarked_ = false;
    deform_ = info_.deformPipeline->Launch(std::move(info), &status, &beginSemantic,
                                           std::move(beforeSubmit));
    if (!deform_) {
        Terminal({}, State::Failed, status);
        return;
    }
    AdvanceDeform();
}
void VulkanSourceWidthJob::AdvanceDeform() {
    // Proof stage, never blocking: Advance checks the proof on the queue
    // timeline. Still running: arm a watch at its value and return. Proven:
    // Advance submitted the apply (beforeSubmit marked the watch); arm it.
    if (info_.completionService && base_->pointCount()) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    if (deform_->ProofValue()) {
        deformMarked_ = false;
        DeformSemantic semantic = DeformSemantic::Ok;
        VkResult status = deform_->Advance(&semantic);
        if (status == VK_NOT_READY) {
            if (watch_ && (!watch_->MarkPhaseSubmitted() ||
                           !info_.completionService->ArmAt(*watch_, deform_->ProofValue(),
                                                           uint64_t(State::DeformPending)))) {
                watch_.reset();
                RouteFailure();
                return;
            }
            auto expected = stageFrom_;
            state_.compare_exchange_strong(expected, State::DeformPending, std::memory_order_acq_rel);
            return;
        }
        if (status != VK_SUCCESS || semantic != DeformSemantic::Ok) {
            // As the synchronous Begin: a rejected evaluation fails the stage.
            bool const marked = deformMarked_;
            if (watch_) {
                if (!marked)
                    info_.completionService->CancelBeforeSubmit(*watch_);
                watch_.reset();
            }
            Terminal({}, marked ? State::LostProof : State::Failed,
                     status == VK_SUCCESS ? VK_ERROR_INITIALIZATION_FAILED : status, stageFrom_);
            return;
        }
    }
    if (watch_ &&
        (!deformMarked_ || !info_.completionService->Arm(*watch_, uint64_t(State::DeformPending)))) {
        watch_.reset();
        RouteFailure();
        return;
    }
    auto expected = stageFrom_;
    state_.compare_exchange_strong(expected, State::DeformPending, std::memory_order_acq_rel);
    if (!base_->pointCount())
        FinishDeform();
}
void VulkanSourceWidthJob::FinishDeform() {
    // Phase-2 RBF device path (rbfVk hook).
    if (rbfVkDeform_) {
        FinishRbfVkDeform();
        return;
    }
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    if (deform_->ProofValue()) {
        AdvanceDeform();
        return;
    }
    DeformSemantic semantic = DeformSemantic::Ok;
    auto proof = deform_->Poll(&semantic);
    if (proof == VK_NOT_READY) {
        if (info_.completionService)
            FailOnOwner();
        return;
    }
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (!deform_->succeeded() || semantic != DeformSemantic::Ok) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto child =
        VulkanSourceGeneration::WithDeform(base_, *deform_, info_.stageValueVersions[stageIndex_]);
    if (!child) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    FinishStage(std::move(child));
}
void VulkanSourceWidthJob::BeginRbfVkDeform() {
    VkResult status = VK_ERROR_INITIALIZATION_FAILED;
    if (info_.completionService) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    bool phaseMarked = false;
    RbfVkDeformPipeline::BeforeSubmit beforeSubmit;
    if (watch_)
        beforeSubmit = [this, &phaseMarked] {
            phaseMarked = watch_ && watch_->MarkPhaseSubmitted();
            return phaseMarked;
        };
    auto const &stage = info_.stages[stageIndex_];
    DeformPipeline::BeginInfo info;
    info.points = base_->PlaneOwner(UsdGenCurveChannels().points);
    info.curveOffsets = base_->curveCount() ? base_->PlaneOwner(UsdGenCurveChannels().curveOffsets) : nullptr;
    info.rootTargets = base_->SourceFrameOwner(stage.deform.rootTargetPlane);
    info.curveCount = base_->curveCount();
    info.pointCount = base_->pointCount();
    info.restSamples = stage.deform.restSamples;
    info.posedSamples = stage.deform.posedSamples;
    info.sampleCount = stage.deform.sampleCount;
    info.smoothing = 0.0;
    info.mask = {stage.deform.mask, 1, nullptr, 0};
    info.enabled = {stage.enabledLiteral ? 1u : 0u, 1, nullptr, 0};
    info.lockRoots = {stage.deform.lockRoots ? 1u : 0u, 1, nullptr, 0};
    info.groomEnvelope = stage.deform.groomEnvelope;
    auto bind = [&](char const *name, auto &field) {
        if (auto f = PackedField(name)) {
            field.data = f->data;
            field.count = uint32_t(f->count);
            field.domain = uint32_t(f->domain);
        }
    };
    bind("mask", info.mask);
    bind("enabled", info.enabled);
    bind("lockRoots", info.lockRoots);
    DeformSemantic beginSemantic = DeformSemantic::Ok;
    rbfVkDeform_ = info_.rbfVkDeformPipeline->Begin(std::move(info), &status, &beginSemantic,
                                                    std::move(beforeSubmit));
    if (!rbfVkDeform_) {
        if (watch_) {
            if (!phaseMarked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, phaseMarked ? State::LostProof : State::Failed, status);
        return;
    }
    if (watch_ &&
        (!phaseMarked || !info_.completionService->Arm(*watch_, uint64_t(State::DeformPending)))) {
        watch_.reset();
        RouteFailure();
        return;
    }
    auto expected = stageFrom_;
    state_.compare_exchange_strong(expected, State::DeformPending, std::memory_order_acq_rel);
    if (!base_->pointCount())
        FinishRbfVkDeform();
}
void VulkanSourceWidthJob::FinishRbfVkDeform() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    DeformSemantic semantic = DeformSemantic::Ok;
    auto proof = rbfVkDeform_->Poll(&semantic);
    if (proof == VK_NOT_READY) {
        if (info_.completionService)
            FailOnOwner();
        return;
    }
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (!rbfVkDeform_->done()) {
        AdvanceRbfVkDeform();
        return;
    }
    if (!rbfVkDeform_->succeeded() || semantic != DeformSemantic::Ok) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto child = VulkanSourceGeneration::WithDeform(base_, *rbfVkDeform_,
                                                    info_.stageValueVersions[stageIndex_]);
    if (!child) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    FinishStage(std::move(child));
}
void VulkanSourceWidthJob::BeginWidth() {
    auto const &controls = info_.stages[stageIndex_].width;
    VkResult status = VK_ERROR_INITIALIZATION_FAILED;
    if (base_->pointCount() && !base_->PlaneOwner(UsdGenCurveChannels().width)) {
        Terminal({}, State::Failed, status);
        return;
    }
    if (info_.completionService && base_->pointCount()) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) -> VulkanCompletionService::DeliveryResult {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    bool phaseMarked = false;
    WidthPipeline::BeforeSubmit beforeSubmit;
    if (watch_)
        beforeSubmit = [this, &phaseMarked] {
            phaseMarked = watch_ && watch_->MarkPhaseSubmitted();
            return phaseMarked;
        };
    width_ =
        info_.widthPipeline->Begin(base_->PlaneOwner(UsdGenCurveChannels().width), base_->pointCount(), controls.width,
                                   controls.replace ? 1u : 0u, &status, std::move(beforeSubmit));
    if (!width_) {
        if (watch_) {
            if (!phaseMarked) {
                info_.completionService->CancelBeforeSubmit(*watch_);
            }
            watch_.reset();
        }
        // A post-admission failure may have submitted native reads;
        // downstream publication must retain its lifetime domain.
        Terminal({}, phaseMarked ? State::LostProof : State::Failed, status);
        return;
    }
    if (watch_ && !phaseMarked) {
        watch_.reset();
        RouteFailure();
        return;
    }
    if (watch_ && !info_.completionService->Arm(*watch_, uint64_t(State::WidthPending))) {
        watch_.reset();
        RouteFailure();
        return;
    }
    auto expected = stageFrom_;
    state_.compare_exchange_strong(expected, State::WidthPending, std::memory_order_acq_rel);
    if (!base_->pointCount())
        FinishWidth(); // Empty has no native event.
    return;
}
void VulkanSourceWidthJob::FinishWidth() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            Terminal({}, State::LostProof, VK_ERROR_DEVICE_LOST);
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    uint32_t semantic = 0;
    auto proof = width_->Poll(&semantic);
    if (proof == VK_NOT_READY) {
        if (info_.completionService)
            FailOnOwner();
        return;
    }
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (semantic) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto result =
        VulkanSourceGeneration::WithWidth(base_, *width_, info_.stageValueVersions[stageIndex_]);
    if (!result) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    FinishStage(std::move(result));
}
} // namespace usdGen::vulkan

namespace usdGen::vulkan {
void VulkanSourceWidthJob::BeginExpressions() {
    auto const &stage = info_.stages[stageIndex_];
    ExprVkCurveGeometry g{base_->PlaneOwner(UsdGenCurveChannels().points),
                          base_->PlaneOwner(UsdGenCurveChannels().rest),
                          base_->PlaneOwner(UsdGenCurveChannels().width),
                          base_->PlaneOwner(UsdGenCurveChannels().curveOffsets),
                          base_->PlaneOwner(UsdGenCurveChannels().stableIds),
                          base_->curveCount(),
                          base_->pointCount()};
    ExprVkGeometryChannels ch{base_->PlaneOwner(UsdGenCurveChannels().hairT),
                              base_->PlaneOwner(UsdGenCurveChannels().rootUV),
                              base_->SourceFrameOwner(UsdGenCurveChannels().sourceRootN),
                              base_->SourceFrameOwner(UsdGenCurveChannels().sourceRootT),
                              base_->SourceFrameOwner(UsdGenCurveChannels().sourceRootB),
                              base_->PlaneOwner(UsdGenCurveChannels().rootPrim)};
    expressions_ = std::make_unique<ExprVkRun>(
        info_.expressionContextPipeline, info_.expressionEvaluatePipeline, stage.expressionBindings,
        std::move(g), std::move(ch), stage.expressionControls);
    state_.store(State::ExpressionPending, std::memory_order_release);
    AdvanceExpressions();
}
void VulkanSourceWidthJob::AdvanceExpressions() {
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (expressions_->done()) {
        if (!expressions_->succeeded()) {
            Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
            return;
        }
        stageFrom_ = State::ExpressionPending;
        if (info_.stages[stageIndex_].kind != VulkanSourceWidthStage::Kind::Width &&
            !info_.stages[stageIndex_].expressionBindings.empty()) {
            BeginExpressionPack();
            return;
        }
        DispatchStage();
        return;
    }
    VkResult status = VK_SUCCESS;
    bool marked = false;
    if (info_.completionService) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    auto before = [this, &marked] {
        marked = !watch_ || watch_->MarkPhaseSubmitted();
        return marked;
    };
    if (!expressions_->Advance(before, &status)) {
        if (watch_) {
            if (!marked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, marked ? State::LostProof : State::Failed, status);
        return;
    }
    // Groom contexts and zero-size evaluators prove immediately without
    // submission.
    if (watch_ && !marked) {
        info_.completionService->CancelBeforeSubmit(*watch_);
        watch_.reset();
    }
    if (watch_) {
        if (!info_.completionService->Arm(*watch_, uint64_t(State::ExpressionPending))) {
            watch_.reset();
            RouteFailure();
        }
        return;
    }
    FinishExpressions();
}
void VulkanSourceWidthJob::FinishExpressions() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            RouteFailure();
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    VkResult proof = expressions_->Poll();
    if (proof == VK_NOT_READY)
        return;
    if (proof != VK_SUCCESS) {
        Terminal({}, proof == VK_ERROR_VALIDATION_FAILED_EXT ? State::Failed : State::LostProof,
                 proof);
        return;
    }
    if (expressions_->NeedsHostStep()) {
        BeginExpressionHost();
        return;
    }
    AdvanceExpressions();
}
void VulkanSourceWidthJob::BeginExpressionWidth() {
    auto const &stage = info_.stages[stageIndex_];
    auto const &ctx = base_->context();
    ExprVkWidthApplyPipeline::Controls controls;
    controls.widthLiteral = stage.width.width;
    controls.maskLiteral = stage.width.mask;
    controls.replaceLiteral = stage.width.replace ? 1 : 0;
    controls.enabledLiteral = stage.enabledLiteral ? 1u : 0u;
    auto bind = [&](char const *name, uint32_t &dom) {
        auto f = expressions_->Find(std::string_view(name));
        if (!f)
            return std::shared_ptr<const ChargedBuffer>{};
        dom = uint32_t(f->domain);
        return f->data;
    };
    auto width = bind("width", controls.widthDom), mask = bind("mask", controls.maskDom),
         replace = bind("replace", controls.replaceDom),
         enabled = bind("enabled", controls.enabledDom);
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = 257 * sizeof(float);
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult status = VK_SUCCESS;
    auto profile = ChargedBuffer::Create(
        ctx, bi, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Scratch, &status);
    void *mapped = nullptr;
    if (!profile ||
        vkMapMemory(ctx->device(), profile->memory(), 0, bi.size, 0, &mapped) != VK_SUCCESS) {
        Terminal({}, State::Failed, VK_ERROR_OUT_OF_HOST_MEMORY);
        return;
    }
    std::memcpy(mapped, stage.width.profile.data(), bi.size);
    vkUnmapMemory(ctx->device(), profile->memory());
    expressionProfile_ = profile;
    if (info_.completionService && base_->pointCount()) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    bool marked = false;
    auto before = [this, &marked] {
        marked = !watch_ || watch_->MarkPhaseSubmitted();
        return marked;
    };
    expressionWidth_ = info_.expressionWidthPipeline->Begin(
        base_->PlaneOwner(UsdGenCurveChannels().width), expressions_->owners(), width, mask, replace, enabled,
        base_->pointCount(), base_->curveCount(), controls, &status, before, expressionProfile_,
        base_->PlaneOwner(UsdGenCurveChannels().hairT), base_->PlaneOwner(UsdGenCurveChannels().curveOffsets));
    if (!expressionWidth_) {
        if (watch_) {
            if (!marked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, marked ? State::LostProof : State::Failed, status);
        return;
    }
    state_.store(State::ExpressionWidthPending, std::memory_order_release);
    if (watch_) {
        if (!marked ||
            !info_.completionService->Arm(*watch_, uint64_t(State::ExpressionWidthPending))) {
            watch_.reset();
            RouteFailure();
        }
        return;
    }
    FinishExpressionWidth();
}
void VulkanSourceWidthJob::FinishExpressionWidth() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            RouteFailure();
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    uint32_t semantic = 0;
    auto proof = expressionWidth_->Poll(&semantic);
    if (proof == VK_NOT_READY)
        return;
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (semantic) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto child = VulkanSourceGeneration::WithExpressionWidth(base_, *expressionWidth_,
                                                             info_.stageValueVersions[stageIndex_]);
    if (!child) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    FinishStage(std::move(child));
}
} // namespace usdGen::vulkan
namespace usdGen::vulkan {
void VulkanSourceWidthJob::BeginGrow() {
    VkResult status = VK_SUCCESS;
    bool marked = false;
    // Composite Grow submits named topology even for zero curves, so its
    // completion marker is required for empty inputs as well.
    if (info_.completionService) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    auto before = [this, &marked] {
        marked = !watch_ || watch_->MarkPhaseSubmitted();
        return marked;
    };
    auto const &stage = info_.stages[stageIndex_];
    grow_ = info_.growVkPipeline->Begin(base_, stage.grow, stage.growTargetPlane,
                                        info_.stageValueVersions[stageIndex_], &status, before);
    if (!grow_) {
        if (watch_) {
            if (!marked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, marked ? State::LostProof : State::Failed, status);
        return;
    }
    state_.store(State::GrowPending, std::memory_order_release);
    if (watch_) {
        if (!marked || !info_.completionService->Arm(*watch_, uint64_t(State::GrowPending))) {
            watch_.reset();
            RouteFailure();
        }
        return;
    }
    FinishGrow();
}
void VulkanSourceWidthJob::FinishGrow() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            RouteFailure();
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    uint32_t semantic = 0;
    auto proof = grow_->Poll(&semantic);
    if (proof == VK_NOT_READY) {
        // A consumed completion marker covers both native submissions. It
        // cannot be consumed without their fences being proved as well.
        if (info_.completionService)
            FailOnOwner();
        return;
    }
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (semantic || !grow_->succeeded()) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto child = grow_->output();
    if (!child) {
        Terminal({}, State::Failed, VK_ERROR_INITIALIZATION_FAILED);
        return;
    }
    FinishStage(std::move(child));
}
} // namespace usdGen::vulkan
namespace usdGen::vulkan {
ExprVkParameterField const *VulkanSourceWidthJob::PackedField(char const *name) const {
    for (auto const &f : packedFields_) {
        auto value = f.destination.GetString();
        if (value.compare(0, 7, "usdGen:") == 0)
            value.erase(0, 7);
        if (value == name)
            return &f;
    }
    return nullptr;
}
void VulkanSourceWidthJob::BeginExpressionPack() {
    auto const &stage = info_.stages[stageIndex_];
    if (packedIndex_ == stage.expressionBindings.size()) {
        stageFrom_ = State::ExpressionPackPending;
        DispatchStage();
        return;
    }
    auto f = expressions_->Find(stage.expressionBindings[packedIndex_].binding.destination);
    if (!f) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    if (!f->count) {
        packedFields_.push_back(*f);
        ++packedIndex_;
        BeginExpressionPack();
        return;
    }
    VkResult status = VK_SUCCESS;
    bool marked = false;
    if (info_.completionService) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    auto before = [this, &marked] {
        marked = !watch_ || watch_->MarkPhaseSubmitted();
        return marked;
    };
    expressionPack_ = info_.expressionPackPipeline->Begin(
        f->data, uint32_t(f->count * f->components), &status, before);
    if (!expressionPack_) {
        if (watch_) {
            if (!marked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, marked ? State::LostProof : State::Failed, status);
        return;
    }
    state_.store(State::ExpressionPackPending, std::memory_order_release);
    if (watch_) {
        if (!marked ||
            !info_.completionService->Arm(*watch_, uint64_t(State::ExpressionPackPending))) {
            watch_.reset();
            RouteFailure();
        }
        return;
    }
    FinishExpressionPack();
}
void VulkanSourceWidthJob::FinishExpressionPack() {
    if (watch_) {
        if (!watchProofReady_)
            return;
        if (!watch_->Retire()) {
            watch_.reset();
            RouteFailure();
            return;
        }
        watch_.reset();
        watchProofReady_ = false;
    }
    uint32_t semantic = 0;
    auto proof = expressionPack_->Poll(&semantic);
    if (proof == VK_NOT_READY)
        return;
    if (proof != VK_SUCCESS) {
        Terminal({}, State::LostProof, proof);
        return;
    }
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (semantic || !expressionPack_->succeeded()) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto f = *expressions_->Find(
        info_.stages[stageIndex_].expressionBindings[packedIndex_].binding.destination);
    f.data = expressionPack_->output();
    packedFields_.push_back(std::move(f));
    ++packedIndex_;
    expressionPack_.reset();
    BeginExpressionPack();
}
} // namespace usdGen::vulkan
namespace usdGen::vulkan {
void VulkanSourceWidthJob::BeginExpressionHost() {
    if (!info_.preparer) {
        auto result = expressions_->ComputeHostStep();
        if (result == VK_SUCCESS)
            result = expressions_->CommitHostStep();
        if (result != VK_SUCCESS) {
            Terminal({}, State::Failed, result);
            return;
        }
        AdvanceExpressions();
        return;
    }
    auto relay = std::make_shared<PrepareRelay>();
    relay->owner = owner_;
    relay->ticket = owner_->ReserveCommandTicket();
    if (!relay->ticket) {
        Terminal({}, State::Failed, VK_ERROR_OUT_OF_HOST_MEMORY);
        return;
    }
    relay->jobLifetime = shared_from_this();
    relay->terminal = relay_;
    relay->quarantine = std::make_unique<std::shared_ptr<PrepareRelay>>();
    expressionHostRelay_ = relay;
    state_.store(State::ExpressionHostPending, std::memory_order_release);
    auto run = expressions_.get();
    UsdGenExecutionTaskGraph::Job task;
    task.cancellation = cancellation_;
    task.tasks.push_back({{}, [relay, run](auto const &, auto finish) {
                              VkResult result = run->ComputeHostStep();
                              // The ready-task graph requires a final publication
                              // token even when only proved scalar registers were
                              // computed. CommitHostStep remains owner-confined.
                              finish(result == VK_SUCCESS ? UsdGenExecutionPipeline::Publish([] {})
                                                          : UsdGenExecutionPipeline::Publish{},
                                     result == VK_SUCCESS
                                         ? std::exception_ptr{}
                                         : std::make_exception_ptr(
                                               std::runtime_error("expression host step failed")));
                          }});
    task.completion = [relay](auto, std::exception_ptr error) {
        relay->taskError.store(bool(error), std::memory_order_release);
        bool posted = false;
        try {
            posted = relay->owner->PostCommand(
                std::move(relay->ticket),
                [relay] {
                    auto job = relay->jobLifetime;
                    if (job)
                        job->TakeExpressionHost(relay);
                },
                [relay] { relay->Fail(); });
        } catch (...) {
        }
        if (!posted)
            relay->Fail();
    };
    if (!info_.preparer->Submit(std::move(task))) {
        relay->jobLifetime.reset();
        expressionHostRelay_.reset();
        Terminal({}, State::Failed, VK_ERROR_OUT_OF_HOST_MEMORY);
    }
}
void VulkanSourceWidthJob::TakeExpressionHost(std::shared_ptr<PrepareRelay> relay) {
    if (relay != expressionHostRelay_ || !owner_->IsExecutingOwner())
        return;
    relay->jobLifetime.reset();
    expressionHostRelay_.reset();
    if (relay_->settled.load(std::memory_order_acquire))
        return;
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    if (relay->taskError.load(std::memory_order_acquire)) {
        Terminal({}, State::Failed, VK_ERROR_VALIDATION_FAILED_EXT);
        return;
    }
    auto result = expressions_->CommitHostStep();
    if (result != VK_SUCCESS) {
        Terminal({}, State::Failed, result);
        return;
    }
    state_.store(State::ExpressionPending, std::memory_order_release);
    AdvanceExpressions();
}
} // namespace usdGen::vulkan
namespace usdGen::vulkan {
void VulkanSourceWidthJob::AdvanceRbfVkDeform() {
    if (Suppressed()) {
        Terminal({}, State::Superseded, VK_SUCCESS);
        return;
    }
    VkResult status = VK_SUCCESS;
    bool marked = false;
    if (info_.completionService) {
        auto self = shared_from_this();
        watch_ = info_.completionService->Reserve(
            self,
            [self](VkResult proof) {
                if (proof == VK_SUCCESS) {
                    self->watchProofReady_ = true;
                    return self->NotifyCompletion()
                               ? VulkanCompletionService::DeliveryResult::Posted
                               : VulkanCompletionService::DeliveryResult::Stale;
                }
                self->NotifyFailure(proof);
                return VulkanCompletionService::DeliveryResult::LostProof;
            },
            [self](VkResult proof) { self->NotifyFailure(proof); });
        if (!watch_) {
            RouteFailure();
            return;
        }
    }
    auto before = [this, &marked] {
        marked = !watch_ || watch_->MarkPhaseSubmitted();
        return marked;
    };
    if (!rbfVkDeform_->Advance(before, &status)) {
        if (watch_) {
            if (!marked)
                info_.completionService->CancelBeforeSubmit(*watch_);
            watch_.reset();
        }
        Terminal({}, marked ? State::LostProof : State::Failed, status);
        return;
    }
    if (watch_ && !marked) {
        info_.completionService->CancelBeforeSubmit(*watch_);
        watch_.reset();
    }
    if (watch_) {
        if (!info_.completionService->Arm(*watch_, uint64_t(State::DeformPending))) {
            watch_.reset();
            RouteFailure();
        }
        return;
    }
    FinishRbfVkDeform();
}
} // namespace usdGen::vulkan
