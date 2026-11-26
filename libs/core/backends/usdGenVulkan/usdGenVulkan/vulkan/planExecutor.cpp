// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "planExecutor.h"
#include "csourceVkCapture.h"
#include "usdGenExpr/vulkan/exprVkPlan.h"
#include "usdGenExpr/vulkan/exprVkStructural.h"
#include "growVkPlan.h"
#include "usdGen/curveLoader.h"
#include "usdGenCurves/types/curveDeviceLayout.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/token.h"
#include "usdGen/vulkan/vkStages.h"
#include <cstring>

#include <algorithm>
#include <cmath>
#include <new>
#include <stdexcept>
#include <utility>
#include <vector>

namespace usdGen::vulkan {
namespace {

// The parameter token for an evaluated destination: the destination itself
// without "usdGen:", else the op's declared spelling of local.
TfToken ParamToken(UsdGenOpNode const &node, TfToken const &destination,
                   std::string const &local) {
    if (destination.GetString() == local)
        return destination;
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    UsdGenOpId const id = ops.Find(node.type);
    if (id != UsdGenOpId::Invalid)
        for (TfToken const &name : ops.Record(id).ParamNames())
            if (name.GetString() == local)
                return name;
    return UsdGenInternToken(local);
}

std::exception_ptr Failure(char const *message) noexcept {
    try {
        return std::make_exception_ptr(std::runtime_error(message));
    } catch (...) {
        return std::current_exception();
    }
}

// This is source capture only.  The loader operates entirely on the immutable
// descriptor snapshot and performs the canonical C3/root/named-plane decode;
// it never evaluates a CPU operator or consults a live stage.
bool Capture(VulkanSourceWidthPlan const &plan, uint64_t topologyVersion,
             uint64_t sourceValueVersion, UsdGenCurveBuffer *out) {
    try {
        if (!out || !plan.Descriptor())
            return false;
        auto const node =
            std::find_if(plan.Descriptor()->nodes.begin(), plan.Descriptor()->nodes.end(),
                         [&](UsdGenOpNode const &candidate) {
                             return candidate.key == plan.SourceNodeKey();
                         });
        if (node == plan.Descriptor()->nodes.end())
            return false;
        if (!node->enabled) {
            UsdGenCurveBuffer empty;
            empty.topologyVersion = topologyVersion;
            empty.valueVersion = sourceValueVersion;
            empty.cvOffsets = {0};
            *out = std::move(empty);
            return true;
        }
        // CurveLoader handles immutable source decoding only. The compiler
        // lowered source resampling to a native topology stage.
        auto captureNode = *node;
        TfToken const &resampleTo =
            UsdGenVulkanKernelFor(UsdGenVulkanStage::CurveSource).Param(vulkanP::kCsResampleTo);
        for (auto &param : captureNode.params)
            if (param.name == resampleTo)
                param.value = VtValue(0);
        UsdGenParamView params{plan.Descriptor().get(), &captureNode};
        UsdGenCaptureContext context;
        context.desc = plan.Descriptor().get();
        context.params = &params;
        context.seed = static_cast<uint32_t>(node->seed);
        context.upstreamGeneration = topologyVersion;
        UsdGenDiagnostics diagnostics;
        UsdGenCurveBuffer result;
        if (plan.SourceControls().surfaceCage) {
            if (!CaptureVulkanSurfaceCage(*plan.Descriptor(), *node, plan.Source(),
                                          plan.SourceControls().maxCurves,
                                          plan.SourceControls().maxCvs, &result, &diagnostics))
                return false;
        } else if (IsVulkanCapturedScatter(*plan.Descriptor(), plan.Source())) {
            if (!CaptureVulkanScatterSource(*plan.Descriptor(), plan.Source(), topologyVersion, sourceValueVersion,
                                            &result))
                return false;
        } else if (!UsdGenCurveLoader::Load(context, plan.Source(), &result, &diagnostics))
            return false;
        if (std::any_of(plan.Steps().begin(), plan.Steps().end(), [](auto const &stage) {
                return stage.kind == VulkanSourceWidthStage::Kind::Grow;
            }))
            CompleteVulkanGrowSource(&result);
        // Session revisions, not decoder-local versions, are authoritative.
        result.topologyVersion = topologyVersion;
        result.valueVersion = sourceValueVersion;
        *out = std::move(result);
        return true;
    } catch (...) {
        return false;
    }
}

struct Relay final : std::enable_shared_from_this<Relay> {
    std::shared_ptr<VulkanPlanExecutor> executor;
    std::shared_ptr<const VulkanSourceWidthPlan> plan;
    VulkanPlanExecutor::Request request;
    VulkanPlanExecutor::Completion completion;
    VulkanGenerationAdapterDomain::OperationLease operation;
    UsdGenExecutionPipeline::CommandTicket ticket;
    UsdGenExecutionPipeline *owner = nullptr;
    std::atomic<bool> settled{false};
    // Loss of the owner transport has no safe destruction boundary. Keep the
    // complete request/domain/packet graph self-retained after issuing its
    // one thread-neutral terminal failure.
    std::shared_ptr<Relay> quarantine;
    std::atomic<bool> quarantined{false};
    // The worker dispatcher may release its last callback reference after it
    // has returned to the owner. Keep normal teardown owner-confined too.
    std::shared_ptr<Relay> ownerAnchor;

    uint64_t TopologyVersion() const noexcept {
        return request.authoritativeRevisions ? request.authoritativeRevisions->topologyVersion
                                              : plan->TopologyVersion();
    }
    uint64_t SourceValueVersion() const noexcept {
        return request.authoritativeRevisions ? request.authoritativeRevisions->sourceValueVersion
                                              : plan->ValueVersion();
    }
    uint64_t FinalValueVersion() const noexcept {
        return request.authoritativeRevisions ? request.authoritativeRevisions->finalValueVersion
                                              : plan->ValueVersion() + 1;
    }

    UsdGenGraphSpec structuralDescriptor;
    std::vector<size_t> structuralNodes;
    size_t structuralCursor = 0;
    bool sourceResampleDisabled = false;
    std::map<UsdGenNodeKey, std::vector<float>> capturedRootTargets;
    bool BeginCapture() {
        auto relay = shared_from_this();
        UsdGenExecutionTaskGraph::Job task;
        task.cancellation = relay->request.cancellation;
        task.tasks.push_back(
            {{}, [relay](auto const &, auto finish) {
                 UsdGenCurveBuffer source;
                 if (!Capture(*relay->plan, relay->TopologyVersion(), relay->SourceValueVersion(),
                              &source)) {
                     finish({}, Failure("Vulkan plan source capture failed"));
                     return;
                 }
                 if (relay->plan->SourceControls().surfaceCage) {
                     for (auto const &stage : relay->plan->Steps()) {
                         if (stage.disabled || stage.kind != VulkanSourceWidthStage::Kind::Deform)
                             continue;
                         auto const &nodes = relay->plan->Descriptor()->nodes;
                         auto node = std::find_if(nodes.begin(), nodes.end(), [&](auto const &n) {
                             return n.key == stage.key;
                         });
                         UsdGenDiagnostics diagnostics;
                         std::vector<float> targets;
                         if (node == nodes.end() ||
                             !GatherVulkanSourceRootTargets(*relay->plan->Descriptor(), *node,
                                                            source, &targets, &diagnostics)) {
                             finish({}, Failure("Vulkan dense source root target capture failed"));
                             return;
                         }
                         relay->capturedRootTargets.emplace(stage.key, std::move(targets));
                     }
                 }
                 finish([relay, source = std::move(
                                    source)]() mutable { relay->AdoptOnOwner(std::move(source)); },
                        {});
             }});
        task.completion = [relay](UsdGenExecutionPipeline::Publish publish,
                                  std::exception_ptr error) mutable {
            // The outstanding OperationLease owns the pipeline through this
            // return handoff. Do not copy that shared owner into the worker:
            // a raw pointer is valid until the accepted owner command retires
            // the lease, and external Close must Drain that owner first.
            auto *owner = relay->owner;
            bool posted = false;
            auto ticket = std::move(relay->ticket);
            try {
                posted =
                    owner &&
                    owner->PostCommand(
                        std::move(ticket),
                        [relay, publish = std::move(publish), error]() mutable {
                            if (error) {
                                relay->FailOnOwner(error);
                                return;
                            }
                            if (publish)
                                publish();
                            else
                                relay->FailOnOwner(
                                    Failure("Vulkan plan source task returned no packet"));
                        },
                        [relay] {
                            relay->FailThreadNeutral(Failure("Vulkan plan owner return cancelled"));
                        });
            } catch (...) {
            }
            if (!posted)
                relay->FailThreadNeutral(Failure("Vulkan plan lost owner transport"));
        };
        return executor->createInfo().preparer->Submit(std::move(task));
    }
    void BeginStructural() {
        if (request.cancellation.Superseded() || executor->IsShutdown()) {
            FailOnOwner(Failure("Vulkan structural expression superseded"));
            return;
        }
        if (structuralCursor == structuralNodes.size()) {
            auto handle = CompileVulkanSourceWidthPlan(structuralDescriptor);
            if (!handle) {
                FailOnOwner(Failure("Vulkan evaluated structural controls are invalid"));
                return;
            }
            auto resolved =
                std::static_pointer_cast<const VulkanSourceWidthPlan>(handle->Payload());
            if (!resolved || resolved->Steps().size() != plan->Steps().size()) {
                FailOnOwner(Failure("Vulkan structural topology plan mismatch"));
                return;
            }
            plan = std::move(resolved);
            if (!BeginCapture()) {
                FailOnOwner(Failure("Vulkan source capture worker admission failed"));
            }
            return;
        }
        size_t index = structuralNodes[structuralCursor];
        auto &node = structuralDescriptor.nodes[index];
        UsdGenOpNode structural = node;
        structural.expressionBindings.clear();
        auto local = [](TfToken const &name) {
            auto text = name.GetString();
            if (text.compare(0, 7, "usdGen:") == 0)
                text.erase(0, 7);
            return text;
        };
        for (auto const &b : node.expressionBindings) {
            auto name = local(b.destination);
            if (UsdGenVulkanKernelOf(node.type).Is(UsdGenVulkanStage::CurveSource) ||
                name == "rbfSamples")
                structural.expressionBindings.push_back(b);
        }
        std::vector<ExprVkCompiledBinding> compiled;
        if (ExprVkCompileNodeBindings(structuralDescriptor, structural, true, true, &compiled) !=
            ExprVkCompileStatus::Ok) {
            FailOnOwner(Failure("Vulkan structural expression compilation failed"));
            return;
        }
        EvaluateExprVkStructural(
            executor->createInfo().expressionContextPipeline,
            executor->createInfo().expressionEvaluatePipeline, operation.CompletionService(),
            std::move(compiled),
            ExprVkPlanControls(structuralDescriptor, node.seed, expr::Domain::Groom),
            [self = shared_from_this(), index, local](ExprVkStructuralValues values,
                                                      VkResult status) {
                if (!self->owner || !self->owner->IsExecutingOwner()) {
                    self->FailThreadNeutral(
                        Failure("Vulkan structural expression lost owner transport"));
                    return;
                }
                try {
                    if (status != VK_SUCCESS) {
                        self->FailOnOwner(
                            Failure("Vulkan structural expression evaluation failed"));
                        return;
                    }
                    auto &node = self->structuralDescriptor.nodes[index];
                    for (auto const &item : values) {
                        auto name = local(item.first);
                        VtValue value;
                        if (name == "useRest") {
                            if (item.second > 1) {
                                self->FailOnOwner(Failure("Vulkan invalid connected useRest"));
                                return;
                            }
                            value = VtValue(item.second != 0);
                        } else {
                            int32_t integer;
                            std::memcpy(&integer, &item.second, 4);
                            if (name == "resampleTo" && (integer < 0 || integer == 1)) {
                                self->FailOnOwner(Failure("Vulkan invalid connected resampleTo"));
                                return;
                            }
                            if (name == "resampleTo" && integer == 0) {
                                self->sourceResampleDisabled = true;
                                integer = 2;
                            }
                            value = VtValue(int(integer));
                        }
                        auto found =
                            std::find_if(node.params.begin(), node.params.end(),
                                         [&](auto const &p) { return local(p.name) == name; });
                        if (found == node.params.end())
                            node.params.push_back({ParamToken(node, item.first, name), value});
                        else
                            found->value = value;
                    }
                    node.expressionBindings.erase(
                        std::remove_if(
                            node.expressionBindings.begin(), node.expressionBindings.end(),
                            [&](auto const &b) { return values.count(b.destination) != 0; }),
                        node.expressionBindings.end());
                    ++self->structuralCursor;
                    self->BeginStructural();
                } catch (...) {
                    self->FailOnOwner(std::current_exception());
                }
            });
    }

    void RetainForLostTransport() noexcept {
        // The flag elects the one writer; nothing reads the cycle.
        if (quarantined.exchange(true, std::memory_order_acq_rel)) return;
        try { quarantine = shared_from_this(); } catch (...) {}
    }

    void FailThreadNeutral(std::exception_ptr error) noexcept {
        if (settled.exchange(true, std::memory_order_acq_rel))
            return;
        // No native submission has occurred on this route.  A broken owner
        // transport may report only this terminal failure from its producer
        // thread; it must not attempt queue work or retire an unproved lease.
        RetainForLostTransport();
        try {
            completion({}, error ? error : Failure("Vulkan plan lost owner transport"));
        } catch (...) {
        }
    }

    void ReleaseOnOwner() noexcept {
        // Break every executor/request/context edge from the serialized owner
        // after delivering normal terminal completion.  The explicit anchor
        // prevents any of these from becoming last-destroyed by a worker.
        ticket = {};
        request = {};
        plan.reset();
        executor.reset();
        completion = {};
        ownerAnchor.reset();
    }

    void FailOnOwner(std::exception_ptr error) noexcept {
        if (settled.exchange(true, std::memory_order_acq_rel))
            return;
        try {
            completion({}, error ? error : Failure("Vulkan plan execution failed"));
        } catch (...) {
        }
        if (!operation.Retire()) {
            RetainForLostTransport();
            return;
        }
        ReleaseOnOwner();
    }
    void AdoptOnOwner(UsdGenCurveBuffer source) noexcept {
        if (settled.load(std::memory_order_acquire))
            return;
        if (executor->IsShutdown() || request.cancellation.Superseded()) {
            FailOnOwner(Failure("Vulkan plan execution superseded"));
            return;
        }
        try {
            VulkanSourceGenerationCreateInfo capture;
            capture.context = request.context;
            capture.source = std::move(source);
            capture.geometry.topologyVersion = TopologyVersion();
            capture.geometry.valueVersion = SourceValueVersion();
            // Match C3 capture space policy, not the presence of rest arrays.
            // The source loader preserves current points in either mode.
            capture.geometry.alreadyDeformed = !plan->SourceControls().useRest;
            capture.geometry.layoutMetadata = UsdGenCurveLayout(UsdGenCurveLayoutType::Cubic,
                                                  (!plan->SourceControls().surfaceCage &&
                                                                  plan->Source().basis == UsdGenTokens().catmullRom
                                                              ? UsdGenCurveLayoutBasis::CatmullRom
                                                              : UsdGenCurveLayoutBasis::BSpline),
                                                  UsdGenCurveLayoutWrap::Pinned);
            VulkanPublicationJob::CreateInfo jobInfo;
            jobInfo.domain = executor->createInfo().domain;
            jobInfo.generation = request.generation;
            jobInfo.tool = request.tool;
            jobInfo.nativeJob.source = std::move(capture);
            jobInfo.nativeJob.widthPipeline = executor->createInfo().widthPipeline;
            jobInfo.nativeJob.expressionContextPipeline =
                executor->createInfo().expressionContextPipeline;
            jobInfo.nativeJob.expressionEvaluatePipeline =
                executor->createInfo().expressionEvaluatePipeline;
            jobInfo.nativeJob.expressionWidthPipeline =
                executor->createInfo().expressionWidthPipeline;
            jobInfo.nativeJob.expressionPackPipeline =
                executor->createInfo().expressionPackPipeline;
            jobInfo.nativeJob.width = plan->Width().width;
            jobInfo.nativeJob.replace = plan->Width().replace ? 1u : 0u;
            jobInfo.nativeJob.stages = plan->Steps();
            for (auto &stage : jobInfo.nativeJob.stages) {
                double const frame = request.evaluationFrame.value_or(plan->Descriptor()->time);
                double const rate = plan->Descriptor()->timeCodesPerSecond > 0
                                        ? plan->Descriptor()->timeCodesPerSecond
                                        : 24.0;
                stage.expressionControls.frame = frame;
                stage.expressionControls.time = frame / rate;
                auto targets = capturedRootTargets.find(stage.key);
                if (targets != capturedRootTargets.end()) {
                    stage.deform.rootTargets = std::move(targets->second);
                    auto const &ids = jobInfo.nativeJob.source.source.curveId;
                    stage.deform.rootStableIds.assign(ids.begin(), ids.end());
                }
            }
            if (sourceResampleDisabled && !jobInfo.nativeJob.stages.empty()) {
                jobInfo.nativeJob.stages.front().disabled = true;
                jobInfo.nativeJob.stages.front().publishTopologyRevision = true;
            }
            if (request.authoritativeRevisions)
                jobInfo.nativeJob.stageValueVersions =
                    request.authoritativeRevisions->intermediateValueVersions;
            jobInfo.nativeJob.stageValueVersions.push_back(FinalValueVersion());
            jobInfo.nativeJob.widthBlendPipeline = executor->createInfo().widthBlendPipeline;
            jobInfo.nativeJob.nonWidthComparePipeline =
                executor->createInfo().nonWidthComparePipeline;
            jobInfo.nativeJob.noisePipeline = executor->createInfo().noisePipeline;
            jobInfo.nativeJob.deformPipeline = executor->createInfo().deformPipeline;
            jobInfo.nativeJob.resampleVkPipeline = executor->createInfo().resampleVkPipeline;
            jobInfo.nativeJob.growVkPipeline = executor->createInfo().growVkPipeline;
            jobInfo.nativeJob.tilesPipeline = executor->createInfo().tilesPipeline;
            jobInfo.nativeJob.boundsPipeline = executor->createInfo().boundsPipeline;
            jobInfo.nativeJob.tileOrder = executor->createInfo().tileOrder;
            jobInfo.nativeJob.tileChunkSize = executor->createInfo().tileChunkSize;
            jobInfo.nativeJob.tileTarget = executor->createInfo().tileTarget;
            jobInfo.nativeJob.rbfVkDeformPipeline = executor->createInfo().rbfVkDeformPipeline;
            jobInfo.nativeJob.lengthCompactionPipeline =
                executor->createInfo().lengthCompactionPipeline;
            if (plan->HasLength()) {
                jobInfo.nativeJob.lengthPipeline = executor->createInfo().lengthPipeline;
                jobInfo.nativeJob.lengthCompactionPipeline =
                    executor->createInfo().lengthCompactionPipeline;
                jobInfo.nativeJob.lengthFactor = plan->LengthFactor();
            }
            if ((!request.authoritativeRevisions && plan->ValueVersion() == UINT64_MAX) ||
                FinalValueVersion() <= SourceValueVersion()) {
                FailOnOwner(Failure("Vulkan plan source value version overflow"));
                return;
            }
            // Width is a distinct immutable child value, never an alias of
            // the decoded source packet's version.
            jobInfo.nativeJob.valueVersion = FinalValueVersion();
            // Publication requires a non-null request anchor. Retaining this
            // relay also retains the caller's optional requestLifetime until
            // its terminal boundary, without making that public input
            // artificially mandatory.
            jobInfo.nativeJob.requestLifetime = shared_from_this();
            jobInfo.nativeJob.completionService = operation.CompletionService();
            jobInfo.nativeJob.preparer = executor->createInfo().preparer;
            std::string reason;
            auto job = VulkanPublicationJob::Create(std::move(jobInfo), &reason);
            if (!job) {
                FailOnOwner(Failure("Vulkan plan publication creation failed"));
                return;
            }
            // Keep the pre-capture lease live until publication has acquired
            // its own native operation lease.  This closes the otherwise
            // observable Close/admission gap between decode and Start.
            auto owner = operation.QueueOwner();
            auto terminal = [self = shared_from_this()](
                                std::shared_ptr<const UsdGenDeviceGeneration> generation,
                                std::exception_ptr error) noexcept {
                if (self->settled.exchange(true, std::memory_order_acq_rel))
                    return;
                try {
                    self->completion(std::move(generation), error);
                } catch (...) {
                }
                if (self->owner && self->owner->IsExecutingOwner())
                    self->ReleaseOnOwner();
                else
                    self->RetainForLostTransport();
            };
            if (!job->Start(*owner, request.cancellation, std::move(terminal))) {
                // Start only rejects before native admission. Completion is
                // still owed by this accepted executor request.
                if (!settled.exchange(true, std::memory_order_acq_rel))
                    try {
                        completion({}, Failure("Vulkan plan publication admission failed"));
                    } catch (...) {
                    }
                operation.CancelBeforeSubmit();
                ReleaseOnOwner();
                return;
            }
            if (!operation.Retire()) {
                job->SuppressPublication();
                FailOnOwner(Failure("Vulkan plan lost owner proof"));
            }
        } catch (...) {
            FailOnOwner(std::current_exception());
        }
    }
};
} // namespace

VulkanPlanExecutor::VulkanPlanExecutor(CreateInfo info) noexcept : info_(std::move(info)) {}

std::shared_ptr<VulkanPlanExecutor> VulkanPlanExecutor::Create(CreateInfo info,
                                                               std::string *reason) {
    if (!info.domain || !info.widthPipeline || !info.preparer) {
        if (reason)
            *reason = "Vulkan plan executor requires injected domain, width "
                      "pipeline, and preparer";
        return {};
    }
    try {
        return std::shared_ptr<VulkanPlanExecutor>(new VulkanPlanExecutor(std::move(info)));
    } catch (...) {
        if (reason)
            *reason = "Vulkan plan executor allocation failed";
        return {};
    }
}

bool VulkanPlanExecutor::Submit(std::shared_ptr<const VulkanSourceWidthPlan> plan, Request request,
                                Completion completion) {
    if (!plan || !request.context || !completion || request.capabilityVersion != 1 ||
        IsShutdown() || request.cancellation.Superseded() ||
        (request.evaluationFrame && !std::isfinite(*request.evaluationFrame)) ||
        (plan->IntermediateCount() && !request.authoritativeRevisions) ||
        (request.authoritativeRevisions &&
         !request.authoritativeRevisions->IsStrictlyOrdered(plan->IntermediateCount())))
        return false;
    for (size_t stageIndex = 0; stageIndex != plan->Steps().size(); ++stageIndex) {
        auto const &stage = plan->Steps()[stageIndex];
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
        bool const invalidEnvelope = !std::isfinite(stage.lengthBlend) || stage.lengthBlend < 0 ||
                                     stage.lengthBlend > 1 ||
                                     !std::isfinite(stage.lengthMaskAmount) ||
                                     stage.lengthMaskAmount < 0 || stage.lengthMaskAmount > 1;
        if ((stage.kind == VulkanSourceWidthStage::Kind::LengthScale ||
             stage.kind == VulkanSourceWidthStage::Kind::LengthCull) &&
            (invalidLengthMethod || invalidLengthMode || invalidLengthRebuild || invalidMinimum ||
             invalidRandom || invalidEnvelope))
            return false;
        if (stage.kind == VulkanSourceWidthStage::Kind::LengthScale &&
            (!info_.lengthPipeline || info_.lengthPipeline->context() != request.context))
            return false;
        if (stage.kind == VulkanSourceWidthStage::Kind::LengthScale && stage.lengthDeviceControls &&
            !info_.lengthPipeline->HasDeviceV1())
            return false;
        if (stage.kind == VulkanSourceWidthStage::Kind::LengthScale &&
            !stage.lengthDeviceControls) {
            bool const literalV1 = stage.randomLo != 1.0f || stage.randomHi != 1.0f;
            bool const nonneutralEnvelope =
                stage.lengthBlend != 1.0f || stage.lengthMaskAmount != 1.0f;
            bool const hasEnvelope = info_.lengthPipeline->HasEnvelopeV1();
            bool const missingEnvelope = nonneutralEnvelope && !hasEnvelope;
            bool const missingLiteral =
                !nonneutralEnvelope && literalV1 && !info_.lengthPipeline->HasLiteralV1();
            bool const missingMinimum = !nonneutralEnvelope && !literalV1 &&
                                        stage.minRemainingLength > 0 &&
                                        !info_.lengthPipeline->HasMinimum();
            bool const missingReparam =
                !nonneutralEnvelope && !literalV1 && stage.minRemainingLength == 0 &&
                stage.lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend &&
                stage.lengthRebuild == VulkanSourceWidthStage::LengthRebuild::Reparam &&
                !info_.lengthPipeline->HasReparam();
            bool const missingCut =
                !nonneutralEnvelope && !literalV1 && stage.minRemainingLength == 0 &&
                stage.lengthMethod == VulkanSourceWidthStage::LengthMethod::CutExtend &&
                stage.lengthRebuild != VulkanSourceWidthStage::LengthRebuild::Reparam &&
                !info_.lengthPipeline->HasCutExtend();
            bool const missingSet =
                !nonneutralEnvelope && !literalV1 && stage.minRemainingLength == 0 &&
                stage.lengthMethod != VulkanSourceWidthStage::LengthMethod::CutExtend &&
                stage.lengthMode == VulkanSourceWidthStage::LengthMode::Set &&
                !info_.lengthPipeline->HasSet();
            if (!std::isfinite(stage.factor) || stage.factor < 0 ||
                (stage.lengthCullOnly && !nonneutralEnvelope) || missingEnvelope ||
                missingLiteral || missingMinimum || missingReparam || missingCut || missingSet)
                return false;
        }
        if (stage.lengthKeepInput &&
            (!info_.lengthCompactionPipeline || !info_.lengthCompactionPipeline->HasKeep()))
            return false;
        if (stage.kind == VulkanSourceWidthStage::Kind::LengthCull &&
            (!info_.lengthCompactionPipeline ||
             info_.lengthCompactionPipeline->context() != request.context ||
             !std::isfinite(stage.factor) || stage.factor < 0 ||
             !std::isfinite(stage.cullThreshold) || stage.cullThreshold < 0 ||
             stage.lengthCullOnly))
            return false;
        if (stage.kind == VulkanSourceWidthStage::Kind::LengthCull &&
            (stage.lengthBlend != 1.0f || stage.lengthMaskAmount != 1.0f) &&
            (!info_.lengthPipeline || info_.lengthPipeline->context() != request.context))
            return false;
        if (stage.kind == VulkanSourceWidthStage::Kind::LengthCull &&
            (stage.lengthBlend != 1.0f || stage.lengthMaskAmount != 1.0f) &&
            !info_.lengthPipeline->HasEnvelopeV1())
            return false;
        if (stage.kind == VulkanSourceWidthStage::Kind::LengthCull &&
            (stage.lengthBlend != 1.0f || stage.lengthMaskAmount != 1.0f)) {
            if (stage.input == 0 || stage.input > stageIndex)
                return false;
            auto const &producer = plan->Steps()[stage.input - 1];
            bool const producerEnvelope =
                producer.lengthBlend != 1.0f || producer.lengthMaskAmount != 1.0f;
            if (producer.kind != VulkanSourceWidthStage::Kind::LengthScale || !producerEnvelope ||
                producer.lengthBlend != stage.lengthBlend ||
                producer.lengthMaskAmount != stage.lengthMaskAmount)
                return false;
        }
        if (stage.requiresNonWidthProof &&
            (!info_.nonWidthComparePipeline ||
             info_.nonWidthComparePipeline->context() != request.context))
            return false;
        if (stage.kind == VulkanSourceWidthStage::Kind::WidthBlend &&
            (!info_.widthBlendPipeline || info_.widthBlendPipeline->context() != request.context))
            return false;
        if (stage.kind == VulkanSourceWidthStage::Kind::Noise &&
            (!info_.noisePipeline || info_.noisePipeline->context() != request.context ||
             !std::isfinite(stage.noise.magnitude) || stage.noise.magnitude < 0 ||
             !std::isfinite(stage.noise.frequency) || stage.noise.frequency <= 0 ||
             !std::isfinite(stage.noise.correlation) || stage.noise.correlation < 0 ||
             stage.noise.correlation > 1 || stage.noise.octaves < 1 || stage.noise.octaves > 6 ||
             !std::isfinite(stage.noise.lacunarity) || stage.noise.lacunarity <= 1 ||
             !std::isfinite(stage.noise.gain) || stage.noise.gain < 0 || stage.noise.gain > 1 ||
             !std::isfinite(stage.noise.preserveLength) || stage.noise.preserveLength < 0 ||
             stage.noise.preserveLength > 1 || !std::isfinite(stage.noise.mask) ||
             stage.noise.mask < 0 || stage.noise.mask > 1))
            return false;
        if (stage.kind == VulkanSourceWidthStage::Kind::Deform) {
            // Phase-2 RBF device path (rbfVk hook): gap admission.
            if (stage.deform.devicePath || info_.rbfVkDeformPipeline) {
                if (!info_.rbfVkDeformPipeline ||
                    info_.rbfVkDeformPipeline->context() != request.context)
                    return false;
                if (stage.deform.sampleCount < 1 || stage.deform.sampleCount > 46336)
                    return false;
            } else {
                if (!info_.deformPipeline || info_.deformPipeline->context() != request.context)
                    return false;
                if (stage.deform.sampleCount < 4 || stage.deform.sampleCount > 100)
                    return false;
            }
            size_t const n = size_t(stage.deform.sampleCount) * 3;
            if (stage.deform.restSamples.size() != n || stage.deform.posedSamples.size() != n)
                return false;
            if (stage.deform.rootTargets.size() % 3 != 0)
                return false;
            for (float v : stage.deform.restSamples)
                if (!std::isfinite(v))
                    return false;
            for (float v : stage.deform.posedSamples)
                if (!std::isfinite(v))
                    return false;
            for (float v : stage.deform.rootTargets)
                if (!std::isfinite(v))
                    return false;
            if (!std::isfinite(stage.deform.mask) || stage.deform.mask < 0 || stage.deform.mask > 1)
                return false;
            if (!std::isfinite(stage.deform.groomEnvelope) || stage.deform.groomEnvelope < 0 ||
                stage.deform.groomEnvelope > 1)
                return false;
        }
        if (stage.kind == VulkanSourceWidthStage::Kind::Resample &&
            (!info_.resampleVkPipeline || info_.resampleVkPipeline->context() != request.context ||
             stage.resampleTarget < 2))
            return false;
    }
    // Tile finalization is not a stage kind: admit the optional pipelines by
    // context only. Either null disables the tile stage downstream.
    if ((info_.tilesPipeline && info_.tilesPipeline->context() != request.context) ||
        (info_.boundsPipeline && info_.boundsPipeline->context() != request.context))
        return false;
    // Admission is owner-only, as OperationLease is the domain's explicit
    // proof that Close cannot detach the queue while a return is outstanding.
    auto operation = info_.domain->AcquireOperation();
    if (!operation || info_.widthPipeline->context() != request.context ||
        operation.CompletionService()->context() != request.context ||
        operation.QueueOwner().get() != operation.CompletionService()->queueOwner()) {
        operation.CancelBeforeSubmit();
        return false;
    }
    auto owner = operation.QueueOwner();
    auto ticket = owner->ReserveCommandTicket();
    if (!ticket) {
        operation.CancelBeforeSubmit();
        return false;
    }
    std::shared_ptr<Relay> relay;
    try {
        relay = std::make_shared<Relay>();
        relay->executor = shared_from_this();
        relay->plan = std::move(plan);
        relay->request = std::move(request);
        relay->completion = std::move(completion);
        relay->operation = std::move(operation);
        relay->ticket = std::move(ticket);
        relay->owner = owner.get();
        relay->ownerAnchor = relay;
        relay->structuralDescriptor = *relay->plan->Descriptor();
        UsdGenVulkanKernel const deform = UsdGenVulkanKernelFor(UsdGenVulkanStage::Deform);
        if (relay->request.evaluationFrame)
            relay->structuralDescriptor.time = *relay->request.evaluationFrame;
        for (size_t i = 0; i < relay->structuralDescriptor.nodes.size(); ++i) {
            auto const &node = relay->structuralDescriptor.nodes[i];
            bool structural = UsdGenVulkanKernelOf(node.type).Is(UsdGenVulkanStage::CurveSource) &&
                              !node.expressionBindings.empty();
            for (auto const &b : node.expressionBindings)
                structural = structural || deform.Names(vulkanP::kDeformRbfSamples, b.destination);
            if (structural)
                relay->structuralNodes.push_back(i);
        }
        if (!relay->structuralNodes.empty()) {
            if (!info_.expressionContextPipeline || !info_.expressionEvaluatePipeline) {
                relay->ticket = {};
                relay->operation.CancelBeforeSubmit();
                relay->ReleaseOnOwner();
                return false;
            }
            relay->BeginStructural();
        } else if (!relay->BeginCapture()) {
            relay->ticket = {};
            relay->operation.CancelBeforeSubmit();
            relay->ReleaseOnOwner();
            return false;
        }
        return true;
    } catch (...) {
        // Transfer to Relay happens before constructing the task closures.
        // This owner frame can still explicitly roll it back on every
        // allocation failure; never rely on OperationLease destruction.
        if (relay) {
            relay->ticket = {};
            relay->operation.CancelBeforeSubmit();
            relay->ReleaseOnOwner();
        } else
            operation.CancelBeforeSubmit();
        return false;
    }
}

void VulkanPlanExecutor::Shutdown() noexcept { closing_.store(true, std::memory_order_release); }

} // namespace usdGen::vulkan
