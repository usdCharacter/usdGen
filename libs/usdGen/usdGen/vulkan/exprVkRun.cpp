// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "exprVkRun.h"
#include "usdGen/expressions/frontend.h"
#include "usdGen/expressions/irExec.h"
#include <cmath>
#include <cstring>
#include <map>
#include <string_view>
#include "usdGen/vulkan/vkStages.h"
namespace usdGen::vulkan {
namespace {
// A destination's name without "usdGen:"; compares text, never interns.
std::string_view Local(TfToken const &t) {
    std::string_view s = t.GetString();
    if (s.compare(0, 7, "usdGen:") == 0)
        s.remove_prefix(7);
    return s;
}
} // namespace
struct ExprVkRun::State {
    std::shared_ptr<ExprVkContextPipeline> context;
    std::shared_ptr<ExprVkEvaluatePipeline> evaluate;
    std::vector<ExprVkCompiledBinding> bindings;
    ExprVkCurveGeometry geometry;
    ExprVkGeometryChannels channels;
    expr::Context controls;
    size_t binding = 0, step = 0;
    uint32_t n = 0, rawLo = 0, rawCount = 0;
    bool finished = false, failed = false, contextReady = false, finalSegment = false;
    bool hostPending = false;
    std::vector<double> produced;
    ExprVkFieldLayout layout;
    std::vector<size_t> hostSteps;
    std::map<size_t, unsigned> rewrite;
    std::unique_ptr<ExprVkContextPipeline::Candidate> build;
    std::unique_ptr<ExprVkEvaluatePipeline::Candidate> run;
    std::shared_ptr<ChargedBuffer> fieldBuffer, program;
    std::shared_ptr<const ChargedBuffer> ownerMap;
    std::vector<ExprVkParameterField> fields;
    std::vector<std::shared_ptr<const ChargedBuffer>> retained;
    bool Map(std::shared_ptr<const ChargedBuffer> const &b, void **p) {
        return b && vkMapMemory(context->context()->device(), b->memory(), 0, b->sizeBytes(), 0,
                                p) == VK_SUCCESS;
    }
    void Unmap(std::shared_ptr<const ChargedBuffer> const &b) {
        vkUnmapMemory(context->context()->device(), b->memory());
    }
};
ExprVkRun::ExprVkRun(std::shared_ptr<ExprVkContextPipeline> c,
                     std::shared_ptr<ExprVkEvaluatePipeline> e,
                     std::vector<ExprVkCompiledBinding> b, ExprVkCurveGeometry g,
                     ExprVkGeometryChannels ch, expr::Context ctl, bool buildOwners)
    : state_(new State) {
    auto &s = *state_;
    s.context = std::move(c);
    s.evaluate = std::move(e);
    s.bindings = std::move(b);
    s.geometry = std::move(g);
    s.channels = std::move(ch);
    s.controls = ctl;

    ExprVkCompiledBinding owners;
    owners.binding.destination = UsdGenVkTokens().owners;
    owners.binding.domain = expr::Domain::Point;
    owners.binding.destinationShape.scalar = expr::ScalarType::Float32;
    owners.binding.destinationShape.components = 1;
    expr::FrontendOptions options;
    options.domain = expr::Domain::Point;
    options.destination = expr::ScalarType::Float32;
    options.components = 1;
    auto compiled = expr::Frontend::Compile("0", options);
    owners.ir = compiled.program.IR();
    owners.literal = {0};
    if (buildOwners)
        s.bindings.push_back(std::move(owners));
    s.failed = !compiled.ok || !s.context || !s.evaluate ||
               s.context->context() != s.evaluate->context() || !std::isfinite(ctl.frame) ||
               !std::isfinite(ctl.time);
}
ExprVkRun::~ExprVkRun() = default;
bool ExprVkRun::done() const noexcept { return state_->finished || state_->failed; }
bool ExprVkRun::succeeded() const noexcept { return state_->finished && !state_->failed; }
bool ExprVkRun::pending() const noexcept { return bool(state_->run) || bool(state_->build); }
void ExprVkRun::Quarantine() noexcept {
    if (state_->run)
        state_->run->Quarantine();
    if (state_->build)
        state_->build->Quarantine();
}
std::shared_ptr<const ChargedBuffer> ExprVkRun::owners() const noexcept { return state_->ownerMap; }
ExprVkParameterField const *ExprVkRun::Find(TfToken const &t) const {
    for (auto const &f : state_->fields)
        if (Local(f.destination) == Local(t))
            return &f;
    return nullptr;
}
bool ExprVkRun::Advance(std::function<bool()> before, VkResult *result) {
    auto &s = *state_;
    if (result)
        *result = VK_ERROR_INITIALIZATION_FAILED;
    if (done() || pending())
        return false;
    if (s.binding == s.bindings.size()) {
        s.finished = true;
        if (result)
            *result = VK_SUCCESS;
        return true;
    }
    auto const &b = s.bindings[s.binding];
    if (!s.contextReady) {
        auto dom = b.binding.domain;
        size_t count = dom == expr::Domain::Groom       ? 1
                       : dom == expr::Domain::Primitive ? s.geometry.curveCount
                                                        : s.geometry.pointCount;
        if ((dom != expr::Domain::Groom && dom != expr::Domain::Primitive &&
             dom != expr::Domain::Point) ||
            count > UINT32_MAX || b.literal.empty() || b.literal.size() > 4 ||
            b.literal.size() != b.binding.destinationShape.components ||
            !expr::ValidProgram(b.ir)) {
            s.failed = true;
            return false;
        }
        s.n = uint32_t(count);
        s.hostSteps = ExprVkHostSteps(b.ir);
        s.step = 0;
        s.rewrite.clear();
        s.layout = ExprVkLayoutFields(
            dom, count, unsigned(b.literal.size()), unsigned(s.hostSteps.size()),
            !!s.geometry.restPoints, !!s.geometry.widths, !!s.channels.rootUV, !!s.channels.rootN,
            !!s.channels.rootT, !!s.channels.rootB, !!s.channels.rootPrim);
        if (!s.layout.totalDoubles || s.layout.totalDoubles > (uint64_t(UINT32_MAX) >> 3)) {
            s.failed = true;
            return false;
        }
        ExprVkContextStatus status;
        s.build = s.context->Begin(s.geometry, s.channels, dom, s.n, s.layout, &status, result,
                                   std::move(before));
        if (!s.build) {
            s.failed = true;
            return false;
        }
        return true;
    }
    while (s.n == 0 && s.step < s.hostSteps.size())
        ++s.step;
    ExprVkSegmentOutputs outputs;
    size_t end = b.ir.instructions.size();
    s.finalSegment = s.step == s.hostSteps.size();
    if (s.finalSegment) {
        outputs.type = b.binding.destinationShape.scalar;
        outputs.regCount = b.ir.outputCount ? b.ir.outputCount : 1;
        for (unsigned c = 0; c < outputs.regCount; ++c)
            outputs.regs[c] = b.ir.outputCount ? b.ir.output[c] : b.ir.result;
    } else {
        auto const &op = b.ir.instructions[s.hostSteps[s.step]];
        end = s.hostSteps[s.step];
        unsigned lo = op.a, hi = op.a;
        if (op.op == expr::IROp::Call) {
            if (!op.b || unsigned(op.a) + op.b > expr::kExprRegisters) {
                s.failed = true;
                return false;
            }
            hi = op.a + op.b - 1;
        } else if (op.op == expr::IROp::Pow || op.op == expr::IROp::Atan2) {
            lo = std::min(op.a, op.b);
            hi = std::max(op.a, op.b);
        }
        outputs.raw = true;
        outputs.rawBase = uint16_t(lo);
        outputs.regCount = hi - lo + 1;
        s.rawLo = lo;
        s.rawCount = outputs.regCount;
    }
    uint64_t slots = uint64_t(s.n) * outputs.regCount;
    if (slots > UINT32_MAX) {
        s.failed = true;
        return false;
    }
    ExprVkPackOptions options;
    options.domain = b.binding.domain;
    options.frame = s.controls.frame;
    options.time = s.controls.time;
    options.seed = s.controls.seed;
    options.descId = s.controls.descId;
    options.count = s.n;
    options.primitiveCount = uint32_t(s.geometry.curveCount);
    std::vector<ExprVkFieldSlot> table(s.layout.varCount);
    for (unsigned v = 0; v < s.layout.varCount; ++v)
        table[v] = {s.layout.offsets[v], s.layout.counts[v], s.layout.domains[v],
                    s.layout.components[v]};
    std::vector<uint32_t> words;
    if (!ExprVkPackProgram(b.ir, end, s.rewrite, outputs, options, table, &words)) {
        s.failed = true;
        return false;
    }
    VkBufferCreateInfo bi{};
    bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bi.size = words.size() * 4;
    bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    s.program = ChargedBuffer::Create(s.context->context(), bi,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                      UsdGenExecutionResourceKind::Scratch, result);
    void *p = nullptr;
    if (!s.Map(s.program, &p)) {
        s.failed = true;
        return false;
    }
    std::memcpy(p, words.data(), bi.size);
    s.Unmap(s.program);
    s.run = s.evaluate->Begin(s.program, s.fieldBuffer, s.ownerMap, s.n, uint32_t(slots), result,
                              std::move(before));
    if (!s.run) {
        s.failed = true;
        return false;
    }
    return true;
}
VkResult ExprVkRun::Poll() {
    auto &s = *state_;
    uint32_t semantic = UINT32_MAX;
    auto const &b = s.bindings[s.binding];
    if (s.build) {
        auto vk = s.build->Poll(&semantic);
        if (vk != VK_SUCCESS)
            return vk;
        if (semantic) {
            s.failed = true;
            s.build.reset();
            return VK_ERROR_VALIDATION_FAILED_EXT;
        }
        s.fieldBuffer = s.build->fields();
        s.ownerMap = s.build->owners();
        void *p = nullptr;
        if (!s.Map(s.fieldBuffer, &p)) {
            s.failed = true;
            return VK_ERROR_MEMORY_MAP_FAILED;
        }
        std::memcpy(static_cast<unsigned char *>(p) +
                        s.layout.offsets[unsigned(expr::Variable::Value)] * 8,
                    b.literal.data(), b.literal.size() * 8);
        s.Unmap(s.fieldBuffer);
        s.contextReady = true;
        s.build.reset();
        return VK_SUCCESS;
    }
    if (!s.run)
        return VK_ERROR_INITIALIZATION_FAILED;
    auto vk = s.run->Poll(&semantic);
    if (vk != VK_SUCCESS)
        return vk;
    if (semantic) {
        s.failed = true;
        s.run.reset();
        return VK_ERROR_VALIDATION_FAILED_EXT;
    }
    if (s.finalSegment) {
        ExprVkParameterField f;
        f.destination = b.binding.destination;
        f.domain = b.binding.domain;
        f.type = b.binding.destinationShape.scalar;
        f.components = b.ir.outputCount ? b.ir.outputCount : 1;
        f.count = s.n;
        if (s.n)
            f.data = s.run->output();
        s.fields.push_back(std::move(f));
        s.retained.push_back(s.fieldBuffer);
        s.retained.push_back(s.ownerMap);
        s.contextReady = false;
        ++s.binding;
    } else {
        s.hostPending = true;
        return VK_SUCCESS;
    }
    s.run.reset();
    s.program.reset();
    if (s.binding == s.bindings.size())
        s.finished = true;
    return VK_SUCCESS;
}

bool ExprVkRun::NeedsHostStep() const noexcept { return state_->hostPending; }
VkResult ExprVkRun::ComputeHostStep() {
    auto &s = *state_;
    auto const &b = s.bindings[s.binding];
    if (!s.hostPending || !s.run)
        return VK_ERROR_INITIALIZATION_FAILED;
    auto const &op = b.ir.instructions[s.hostSteps[s.step]];
    unsigned need = op.op == expr::IROp::Call                                  ? op.b
                    : (op.op == expr::IROp::Pow || op.op == expr::IROp::Atan2) ? 2
                                                                               : 1;
    std::vector<std::vector<double>> gathered(need, std::vector<double>(s.n));
    std::vector<double const *> inputs(need);
    void *p = nullptr;
    if (!s.Map(s.run->output(), &p)) {

        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    auto bits = static_cast<double const *>(p);
    for (unsigned k = 0; k < need; ++k) {
        unsigned reg = op.op == expr::IROp::Call ? op.a + k : k == 0 ? op.a : op.b;
        for (uint32_t e = 0; e < s.n; ++e)
            gathered[k][e] = bits[e * s.rawCount + reg - s.rawLo];
        inputs[k] = gathered[k].data();
    }
    s.Unmap(s.run->output());
    s.produced.assign(s.n, 0.0);
    if (!ExprVkEvaluateHostStep(op, inputs.data(), s.n, s.produced.data())) {

        return VK_ERROR_INITIALIZATION_FAILED;
    }
    return VK_SUCCESS;
}
VkResult ExprVkRun::CommitHostStep() {
    auto &s = *state_;
    void *p = nullptr;
    if (!s.hostPending || s.produced.size() != s.n)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!s.Map(s.fieldBuffer, &p)) {
        s.failed = true;
        return VK_ERROR_MEMORY_MAP_FAILED;
    }
    std::memcpy(static_cast<unsigned char *>(p) +
                    s.layout.offsets[ExprVkEncoding::kFirstInjectionSlot + s.step] * 8,
                s.produced.data(), s.n * 8);
    s.Unmap(s.fieldBuffer);
    s.rewrite[s.hostSteps[s.step]] = ExprVkEncoding::kFirstInjectionSlot + unsigned(s.step);
    ++s.step;
    s.hostPending = false;
    s.produced.clear();
    s.run.reset();
    s.program.reset();
    return VK_SUCCESS;
}
} // namespace usdGen::vulkan
