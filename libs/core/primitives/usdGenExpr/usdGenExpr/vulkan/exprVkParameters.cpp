// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "exprVkParameters.h"

#include "usdGen/expressions/irExec.h"

#include <cstring>
#include <limits>
#include <map>
#include <string_view>

namespace usdGen::vulkan {
namespace {
constexpr uint64_t kWaitNs = 30000000000ull;

void Error(std::vector<std::string>* diagnostics, std::string const& message) {
    if (diagnostics) diagnostics->push_back(message);
}
// A destination's name without "usdGen:"; compares text, never interns.
std::string_view CanonicalName(UsdGenToken const& name) {
    std::string_view value = name.GetString();
    if (value.compare(0, 7, "usdGen:") == 0) value.remove_prefix(7);
    return value;
}
bool MapWhole(DeviceContext const& context, ChargedBuffer const& buffer, void** mapped) {
    return vkMapMemory(context.device(), buffer.memory(), 0, buffer.sizeBytes(), 0, mapped) ==
        VK_SUCCESS;
}
} // namespace

ExprVkParameterEvaluator::ExprVkParameterEvaluator(
    std::shared_ptr<ExprVkContextPipeline> contextPipeline,
    std::shared_ptr<ExprVkEvaluatePipeline> evaluatePipeline)
    : context_(std::move(contextPipeline)), evaluate_(std::move(evaluatePipeline)) {}
ExprVkParameterEvaluator::~ExprVkParameterEvaluator() = default;

ExprVkParameterField const* ExprVkParameterEvaluator::Find(UsdGenToken const& destination) const {
    auto canonical = CanonicalName(destination);
    for (auto const& field : fields_)
        if (CanonicalName(field.destination) == canonical) return &field;
    return nullptr;
}

ExprVkParameterStatus ExprVkParameterEvaluator::Evaluate(
    std::vector<ExprVkCompiledBinding> const& bindings,
    ExprVkCurveGeometry const& geometry, ExprVkGeometryChannels const& channels,
    expr::Context controls, std::vector<std::string>* diagnostics) {
    fields_.clear();
    retained_.clear();
    if (!context_ || !evaluate_ || context_->context() != evaluate_->context()) {
        Error(diagnostics, "null or mismatched expression pipelines");
        return ExprVkParameterStatus::InvalidArgument;
    }
    auto context = context_->context();
    if (!std::isfinite(controls.frame) || !std::isfinite(controls.time)) {
        Error(diagnostics, "non-finite frame/time");
        return ExprVkParameterStatus::InvalidArgument;
    }
    std::vector<ExprVkParameterField> fields;
    std::vector<std::shared_ptr<void>> retained;
    auto retain = [&](std::shared_ptr<const void> object) {
        retained.push_back(std::const_pointer_cast<void>(object));
    };
    for (auto const& item : bindings) {
        auto fail = [&](ExprVkParameterStatus status, std::string const& message) {
            Error(diagnostics, item.binding.destination.GetString() + ": " + message);
            return status;
        };
        expr::Domain const domain = item.binding.domain;
        size_t const n = domain == expr::Domain::Groom ? 1
            : domain == expr::Domain::Primitive ? geometry.curveCount
                                                 : geometry.pointCount;
        if (domain != expr::Domain::Groom && domain != expr::Domain::Primitive &&
            domain != expr::Domain::Point)
            return fail(ExprVkParameterStatus::InvalidArgument, "invalid evaluation domain");
        if (n > UINT32_MAX || item.literal.size() != item.binding.destinationShape.components ||
            item.literal.empty() || item.literal.size() > 4)
            return fail(ExprVkParameterStatus::InvalidArgument, "literal shape mismatch");
        if (!expr::ValidProgram(item.ir))
            return fail(ExprVkParameterStatus::InvalidArgument, "invalid program");
        std::vector<size_t> const steps = ExprVkHostSteps(item.ir);
        if (steps.size() > 4096)
            return fail(ExprVkParameterStatus::InvalidArgument, "too many host steps");
        ExprVkFieldLayout const layout = ExprVkLayoutFields(domain, n,
            unsigned(item.literal.size()), unsigned(steps.size()), !!geometry.restPoints,
            !!geometry.widths, !!channels.rootUV, !!channels.rootN, !!channels.rootT,
            !!channels.rootB, !!channels.rootPrim);
        if (layout.totalDoubles == 0 || layout.totalDoubles > (uint64_t(UINT32_MAX) >> 3))
            return fail(ExprVkParameterStatus::InvalidArgument, "field layout overflow");
        ExprVkContextStatus contextStatus = ExprVkContextStatus::DeviceError;
        VkResult vk = VK_SUCCESS;
        auto built = context_->Begin(geometry, channels, domain, uint32_t(n), layout,
                                     &contextStatus, &vk);
        if (!built || contextStatus != ExprVkContextStatus::Ok) {
            if (contextStatus == ExprVkContextStatus::InvalidGeometry)
                return fail(ExprVkParameterStatus::InvalidGeometry, "invalid geometry");
            if (contextStatus == ExprVkContextStatus::InvalidChannel)
                return fail(ExprVkParameterStatus::InvalidChannel, "invalid channel");
            return fail(ExprVkParameterStatus::DeviceError, "context build failed");
        }
        uint32_t semantic = UINT32_MAX;
        vk = built->Wait(kWaitNs, &semantic);
        if (vk != VK_SUCCESS) {
            built->Quarantine();
            return fail(ExprVkParameterStatus::DeviceError, "context wait failed");
        }
        if (semantic == 1)
            return fail(ExprVkParameterStatus::InvalidGeometry, "device geometry validation failed");
        if (semantic != 0)
            return fail(ExprVkParameterStatus::InvalidChannel, "device channel validation failed");
        std::shared_ptr<ChargedBuffer> fieldBuffer = built->fields();
        std::shared_ptr<const ChargedBuffer> owners = built->owners();
        if (!fieldBuffer || !owners) return fail(ExprVkParameterStatus::DeviceError,
                                                 "context produced no fields");
        // Upload the $value literal (the context dispatch is proved idle, so
        // no upload can race a dispatch).
        void* mapped = nullptr;
        if (!MapWhole(*context, *fieldBuffer, &mapped))
            return fail(ExprVkParameterStatus::DeviceError, "field map failed");
        uint64_t const literalAt =
            layout.offsets[static_cast<unsigned>(expr::Variable::Value)];
        std::memcpy(static_cast<unsigned char*>(mapped) + literalAt * 8, item.literal.data(),
                    item.literal.size() * 8);
        vkUnmapMemory(context->device(), fieldBuffer->memory());
        std::vector<ExprVkFieldSlot> table(layout.varCount);
        for (unsigned v = 0; v < layout.varCount; ++v) {
            table[v].offsetDoubles = layout.offsets[v];
            table[v].count = layout.counts[v];
            table[v].domain = layout.domains[v];
            table[v].components = layout.components[v];
        }
        ExprVkPackOptions options;
        options.domain = domain;
        options.frame = controls.frame;
        options.time = controls.time;
        options.seed = controls.seed;
        options.descId = controls.descId;
        options.count = uint32_t(n);
        options.primitiveCount = uint32_t(geometry.curveCount);
        auto makeProgram = [&](std::vector<uint32_t> const& words,
                               std::shared_ptr<ChargedBuffer>* out) {
            VkBufferCreateInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size = words.size() * 4;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            VkResult created = VK_SUCCESS;
            *out = ChargedBuffer::Create(context, bi,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                UsdGenExecutionResourceKind::Scratch, &created);
            if (!*out) return created;
            void* programMapped = nullptr;
            if (!MapWhole(*context, **out, &programMapped)) return VK_ERROR_MEMORY_MAP_FAILED;
            std::memcpy(programMapped, words.data(), words.size() * 4);
            vkUnmapMemory(context->device(), (*out)->memory());
            retain(*out);
            return VK_SUCCESS;
        };
        auto runSegment = [&](size_t endExclusive, std::map<size_t, unsigned> const& rewrite,
                              ExprVkSegmentOutputs const& outputs, uint32_t outSlots,
                              std::vector<double>* bits) {
            std::vector<uint32_t> words;
            std::string diagnostic;
            if (!ExprVkPackProgram(item.ir, endExclusive, rewrite, outputs, options, table,
                                   &words, &diagnostic))
                return fail(ExprVkParameterStatus::DeviceError,
                            "pack failed: " + diagnostic);
            std::shared_ptr<ChargedBuffer> program;
            if (makeProgram(words, &program) != VK_SUCCESS)
                return fail(ExprVkParameterStatus::DeviceError, "program upload failed");
            VkResult begun = VK_SUCCESS;
            auto run = evaluate_->Begin(program, fieldBuffer, owners, uint32_t(n), outSlots,
                                        &begun);
            if (!run) return fail(ExprVkParameterStatus::DeviceError, "segment begin failed");
            uint32_t status = UINT32_MAX;
            if (run->Wait(kWaitNs, &status) != VK_SUCCESS) {
                run->Quarantine();
                return fail(ExprVkParameterStatus::DeviceError, "segment wait failed");
            }
            if (status != 0)
                return fail(ExprVkParameterStatus::InvalidValue,
                            "value the destination cannot represent");
            if (outSlots && bits) {
                void* readback = nullptr;
                if (!MapWhole(*context, *run->output(), &readback))
                    return fail(ExprVkParameterStatus::DeviceError, "segment readback failed");
                bits->resize(outSlots);
                std::memcpy(bits->data(), readback, size_t(outSlots) * 8);
                vkUnmapMemory(context->device(), run->output()->memory());
            }
            retain(run->output());
            return ExprVkParameterStatus::Ok;
        };
        // Intermediate segments: dump each host step's input registers, run
        // the step on the host, inject the results for the next prefix.
        std::map<size_t, unsigned> rewrite;
        for (size_t k = 0; k < steps.size(); ++k) {
            expr::IRInstruction const& step = item.ir.instructions[steps[k]];
            unsigned lo = 0, hi = 0;
            if (step.op == expr::IROp::Call) {
                if (step.b == 0 || step.b > expr::kExprRegisters ||
                    unsigned(step.a) + step.b > expr::kExprRegisters)
                    return fail(ExprVkParameterStatus::InvalidArgument, "call block out of range");
                lo = step.a;
                hi = step.a + step.b - 1;
            } else if (step.op == expr::IROp::Pow || step.op == expr::IROp::Atan2) {
                lo = step.a < step.b ? step.a : step.b;
                hi = step.a < step.b ? step.b : step.a;
            } else {
                lo = hi = step.a;
            }
            ExprVkSegmentOutputs outputs;
            outputs.raw = true;
            outputs.rawBase = uint16_t(lo);
            outputs.regCount = hi - lo + 1;
            uint64_t const outSlots = uint64_t(n) * outputs.regCount;
            if (outSlots == 0 || outSlots > uint64_t(UINT32_MAX)) {
                if (n == 0) continue; // No elements: no host inputs to read.
                return fail(ExprVkParameterStatus::InvalidArgument, "segment dump too large");
            }
            std::vector<double> bits;
            ExprVkParameterStatus run =
                runSegment(steps[k], rewrite, outputs, uint32_t(outSlots), &bits);
            if (run != ExprVkParameterStatus::Ok) return run;
            // Gather this step's inputs from the dumped range.
            unsigned const need =
                step.op == expr::IROp::Call ? step.b : step.op == expr::IROp::Pow ||
                        step.op == expr::IROp::Atan2
                    ? 2
                    : 1;
            std::vector<std::vector<double>> gathered(need, std::vector<double>(n));
            std::vector<double const*> inputs(need);
            for (unsigned k2 = 0; k2 < need; ++k2) {
                unsigned const reg = step.op == expr::IROp::Call ? step.a + k2
                    : k2 == 0                                     ? step.a
                                                                  : step.b;
                for (size_t e = 0; e < n; ++e)
                    gathered[k2][e] = bits[e * outputs.regCount + (reg - lo)];
                inputs[k2] = gathered[k2].data();
            }
            std::vector<double> produced(n, 0.0);
            if (!ExprVkEvaluateHostStep(step, inputs.data(), n, produced.data()))
                return fail(ExprVkParameterStatus::DeviceError, "host step refused");
            void* inject = nullptr;
            if (!MapWhole(*context, *fieldBuffer, &inject))
                return fail(ExprVkParameterStatus::DeviceError, "injection map failed");
            uint64_t const at = layout.offsets[ExprVkEncoding::kFirstInjectionSlot + k];
            std::memcpy(static_cast<unsigned char*>(inject) + at * 8, produced.data(), n * 8);
            vkUnmapMemory(context->device(), fieldBuffer->memory());
            rewrite[steps[k]] = ExprVkEncoding::kFirstInjectionSlot + unsigned(k);
        }
        // Final segment: cooked outputs.
        unsigned const outCount =
            item.ir.outputCount ? item.ir.outputCount : 1;
        ExprVkSegmentOutputs finalOutputs;
        finalOutputs.raw = false;
        finalOutputs.regCount = outCount;
        finalOutputs.type = item.binding.destinationShape.scalar;
        for (unsigned c = 0; c < outCount; ++c)
            finalOutputs.regs[c] =
                item.ir.outputCount ? item.ir.output[c] : item.ir.result;
        uint64_t const finalSlots = uint64_t(n) * outCount;
        if (finalSlots > uint64_t(UINT32_MAX))
            return fail(ExprVkParameterStatus::InvalidArgument, "output too large");
        ExprVkParameterStatus cooked =
            runSegment(item.ir.instructions.size(), rewrite, finalOutputs,
                       uint32_t(finalSlots), nullptr);
        if (cooked != ExprVkParameterStatus::Ok) return cooked;
        ExprVkParameterField field;
        field.destination = item.binding.destination;
        field.count = n;
        field.type = item.binding.destinationShape.scalar;
        field.components = outCount;
        field.domain = domain;
        if (n && finalSlots) {
            // The last retained output is this binding's field.
            field.data = std::static_pointer_cast<const ChargedBuffer>(retained.back());
        }
        fields.push_back(std::move(field));
        retain(fieldBuffer);
        retain(owners);
    }
    fields_ = std::move(fields);
    retained_ = std::move(retained);
    return ExprVkParameterStatus::Ok;
}

} // namespace usdGen::vulkan
