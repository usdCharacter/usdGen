// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "lengthScalePipeline.h"
#include "lengthEnvelope.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <new>

namespace usdGen::vulkan {
struct LengthScalePipeline::Native {
    std::shared_ptr<DeviceContext> context;
    bool reparam = false;
    bool minimumFlavor = false;
    bool literalV1 = false;
    bool envelopeV1 = false;
    bool deviceV1 = false;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    ~Native() {
        if (!context) return;
        auto d = context->device();
        if (pipeline) vkDestroyPipeline(d, pipeline, nullptr);
        if (layout) vkDestroyPipelineLayout(d, layout, nullptr);
        if (descriptors) vkDestroyDescriptorSetLayout(d, descriptors, nullptr);
        if (shader) vkDestroyShaderModule(d, shader, nullptr);
    }
};
struct LengthScalePipeline::Candidate::State {
    std::shared_ptr<Native> native;
    // Retain every source and output resource while a native submission is pending.
    std::shared_ptr<const ChargedBuffer> input, offsets, hairT, stableIds;
    std::shared_ptr<ChargedBuffer> output, status, keep;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool pending = false, lost = false, proved = false;
    bool minimumFlavor = false;
    bool literalV1 = false;
    bool envelopeV1 = false;
    bool deviceV1 = false;
    LengthScalePipeline::MinimumControls minimumControls{};
    LengthScalePipeline::LiteralV1Controls literalV1Controls{};
    LengthScalePipeline::EnvelopeV1Controls envelopeV1Controls{};
    LengthScalePipeline::DeviceV1Controls deviceV1Controls{};
    uint32_t points = 0, curves = 0, semantic = UINT32_MAX;
    // Preallocate the self-retention holder so quarantine is noexcept.
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d = native->context->device();
        if (fence) vkDestroyFence(d, fence, nullptr);
        if (commands) vkDestroyCommandPool(d, commands, nullptr);
        if (descriptors) vkDestroyDescriptorPool(d, descriptors, nullptr);
    }
};
LengthScalePipeline::LengthScalePipeline(std::shared_ptr<Native> n):native_(std::move(n)){}
LengthScalePipeline::~LengthScalePipeline()=default;
std::shared_ptr<DeviceContext> const& LengthScalePipeline::context() const noexcept{return native_->context;}
std::shared_ptr<LengthScalePipeline> LengthScalePipeline::Create(std::shared_ptr<DeviceContext> context,std::vector<uint32_t> const& code,VkResult* result){
    return CreateImpl(std::move(context),code,false,result);
}
std::shared_ptr<LengthScalePipeline> LengthScalePipeline::CreateImpl(std::shared_ptr<DeviceContext> context,std::vector<uint32_t> const& code,bool reparam,VkResult* result,bool minimumFlavor,bool literalV1,bool envelopeV1,bool deviceV1){
    auto finish=[&](VkResult r){if(result)*result=r;}; finish(VK_ERROR_INITIALIZATION_FAILED);
    if (!context || code.size() < 5 || code.front() != 0x07230203u) return {};
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(context->physicalDevice(), &p);
    uint32_t const descriptorCount = deviceV1 ? 13u : ((literalV1 || envelopeV1) ? 6u : ((reparam || minimumFlavor) ? 5u : 4u));
    uint32_t const pushBytes = deviceV1 ? 88u : (envelopeV1 ? 56u : (literalV1 ? 48u : (minimumFlavor ? 32u : (reparam ? 20u : 16u))));
    if (p.limits.maxComputeWorkGroupInvocations < 256 || p.limits.maxComputeWorkGroupSize[0] < 256 ||
        p.limits.maxPerStageDescriptorStorageBuffers < descriptorCount ||
        p.limits.maxDescriptorSetStorageBuffers < descriptorCount || p.limits.maxPushConstantsSize < pushBytes) return {};
    try { auto n=std::make_shared<Native>();n->context=std::move(context);n->reparam=reparam;n->minimumFlavor=minimumFlavor;n->literalV1=literalV1;n->envelopeV1=envelopeV1;n->deviceV1=deviceV1;auto d=n->context->device();
        VkShaderModuleCreateInfo sm{};sm.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;sm.codeSize=code.size()*4;sm.pCode=code.data();VkResult r=vkCreateShaderModule(d,&sm,nullptr,&n->shader);if(r!=VK_SUCCESS){finish(r);return{};}
        VkDescriptorSetLayoutBinding b[13]{};for(uint32_t i=0;i<descriptorCount;++i){b[i].binding=i;b[i].descriptorCount=1;b[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;b[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
        VkDescriptorSetLayoutCreateInfo ds{};ds.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;ds.bindingCount=descriptorCount;ds.pBindings=b;r=vkCreateDescriptorSetLayout(d,&ds,nullptr,&n->descriptors);if(r!=VK_SUCCESS){finish(r);return{};}
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,pushBytes};VkPipelineLayoutCreateInfo pl{};pl.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pl.setLayoutCount=1;pl.pSetLayouts=&n->descriptors;pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&push;r=vkCreatePipelineLayout(d,&pl,nullptr,&n->layout);if(r!=VK_SUCCESS){finish(r);return{};}
        VkComputePipelineCreateInfo cp{};cp.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;cp.layout=n->layout;cp.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;cp.stage.module=n->shader;cp.stage.pName="main";r=vkCreateComputePipelines(d,VK_NULL_HANDLE,1,&cp,nullptr,&n->pipeline);if(r!=VK_SUCCESS){finish(r);return{};}
        auto out=std::shared_ptr<LengthScalePipeline>(new LengthScalePipeline(std::move(n)));finish(VK_SUCCESS);return out;
    }catch(std::bad_alloc const&){finish(VK_ERROR_OUT_OF_HOST_MEMORY);return{};}
}
std::shared_ptr<LengthScalePipeline> LengthScalePipeline::CreateWithSet(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& scaleCode,
    std::vector<uint32_t> const& setCode, VkResult* result) {
    auto finish=[&](VkResult r){if(result)*result=r;}; finish(VK_ERROR_INITIALIZATION_FAILED);
    auto scale=Create(std::move(context),scaleCode,result);
    if (!scale) return {};
    // A separate Native retains a distinct shader/layout/pipeline lifetime.
    // In particular, old scale modules remain valid for Create() callers.
    auto set=Create(scale->context(),setCode,result);
    if (!set) return {};
    scale->setNative_=std::move(set->native_);
    finish(VK_SUCCESS);
    return scale;
}
std::shared_ptr<LengthScalePipeline> LengthScalePipeline::CreateWithCutExtend(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& scaleCode,
    std::vector<uint32_t> const& setCode, std::vector<uint32_t> const& cutCode,
    VkResult* result) {
    auto finish=[&](VkResult r){if(result)*result=r;}; finish(VK_ERROR_INITIALIZATION_FAILED);
    auto pipeline=CreateWithSet(std::move(context),scaleCode,setCode,result);
    if (!pipeline) return {};
    // Keep the cut/extend program native and independently owned; its ABI is
    // trusted by the caller just as the optional set program is.
    auto cut=Create(pipeline->context(),cutCode,result);
    if (!cut) return {};
    pipeline->cutNative_=std::move(cut->native_);
    finish(VK_SUCCESS);
    return pipeline;
}
std::shared_ptr<LengthScalePipeline> LengthScalePipeline::CreateWithReparam(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& scaleCode,
    std::vector<uint32_t> const& setCode, std::vector<uint32_t> const& cutCode,
    std::vector<uint32_t> const& reparamCode, VkResult* result) {
    auto finish=[&](VkResult r){if(result)*result=r;}; finish(VK_ERROR_INITIALIZATION_FAILED);
    auto pipeline=CreateWithCutExtend(std::move(context),scaleCode,setCode,cutCode,result);
    if (!pipeline) return {};
    auto reparam=CreateImpl(pipeline->context(),reparamCode,true,result);
    if (!reparam) return {};
    pipeline->reparamNative_=std::move(reparam->native_);
    finish(VK_SUCCESS);
    return pipeline;
}
std::shared_ptr<LengthScalePipeline> LengthScalePipeline::CreateWithMinimum(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& scaleCode,
    std::vector<uint32_t> const& setCode, std::vector<uint32_t> const& cutCode,
    std::vector<uint32_t> const& reparamCode, std::vector<uint32_t> const& minimumCode,
    VkResult* result) {
    auto finish=[&](VkResult r){if(result)*result=r;}; finish(VK_ERROR_INITIALIZATION_FAILED);
    auto pipeline=CreateWithReparam(std::move(context),scaleCode,setCode,cutCode,reparamCode,result);
    if (!pipeline) return {};
    auto minimum=CreateImpl(pipeline->context(),minimumCode,false,result,true);
    if (!minimum) return {};
    pipeline->minimumNative_=std::move(minimum->native_);
    finish(VK_SUCCESS);
    return pipeline;
}
std::shared_ptr<LengthScalePipeline> LengthScalePipeline::CreateWithLiteralV1(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& scaleCode,
    std::vector<uint32_t> const& setCode, std::vector<uint32_t> const& cutCode,
    std::vector<uint32_t> const& reparamCode, std::vector<uint32_t> const& minimumCode,
    std::vector<uint32_t> const& literalV1Code, VkResult* result) {
    auto finish=[&](VkResult r){if(result)*result=r;}; finish(VK_ERROR_INITIALIZATION_FAILED);
    auto pipeline=CreateWithMinimum(std::move(context),scaleCode,setCode,cutCode,reparamCode,minimumCode,result);
    if (!pipeline) return {};
    auto literal=CreateImpl(pipeline->context(),literalV1Code,false,result,false,true);
    if (!literal) return {};
    pipeline->literalV1Native_=std::move(literal->native_);
    finish(VK_SUCCESS);
    return pipeline;
}
std::shared_ptr<LengthScalePipeline> LengthScalePipeline::CreateWithEnvelopeV1(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& scaleCode,
    std::vector<uint32_t> const& setCode, std::vector<uint32_t> const& cutCode,
    std::vector<uint32_t> const& reparamCode, std::vector<uint32_t> const& minimumCode,
    std::vector<uint32_t> const& literalV1Code, std::vector<uint32_t> const& envelopeV1Code,
    VkResult* result) {
    auto finish=[&](VkResult r){if(result)*result=r;}; finish(VK_ERROR_INITIALIZATION_FAILED);
    auto pipeline=CreateWithLiteralV1(std::move(context),scaleCode,setCode,cutCode,reparamCode,minimumCode,literalV1Code,result);
    if (!pipeline) return {};
    auto envelope=CreateImpl(pipeline->context(),envelopeV1Code,false,result,false,false,true);
    if (!envelope) return {};
    pipeline->envelopeV1Native_=std::move(envelope->native_);
    finish(VK_SUCCESS);
    return pipeline;
}
bool LengthScalePipeline::HasSet() const noexcept { return bool(setNative_); }
bool LengthScalePipeline::HasCutExtend() const noexcept { return bool(cutNative_); }
bool LengthScalePipeline::HasReparam() const noexcept { return bool(reparamNative_); }
bool LengthScalePipeline::HasMinimum() const noexcept { return bool(minimumNative_); }
bool LengthScalePipeline::HasLiteralV1() const noexcept { return bool(literalV1Native_); }
bool LengthScalePipeline::HasEnvelopeV1() const noexcept { return bool(envelopeV1Native_); }
LengthScalePipeline::Candidate::Candidate(std::shared_ptr<State> s):state_(std::move(s)){}
std::shared_ptr<LengthScalePipeline> LengthScalePipeline::CreateWithDeviceV1(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& scaleCode,
    std::vector<uint32_t> const& setCode, std::vector<uint32_t> const& cutCode,
    std::vector<uint32_t> const& reparamCode, std::vector<uint32_t> const& minimumCode,
    std::vector<uint32_t> const& literalV1Code, std::vector<uint32_t> const& envelopeV1Code,
    std::vector<uint32_t> const& deviceV1Code, VkResult* result) {
    auto pipeline = CreateWithEnvelopeV1(std::move(context), scaleCode, setCode,
        cutCode, reparamCode, minimumCode, literalV1Code, envelopeV1Code, result);
    if (!pipeline) return {};
    auto connected = CreateImpl(pipeline->context(), deviceV1Code, false, result, false, false, false, true);
    if (!connected) return {};
    pipeline->deviceV1Native_ = std::move(connected->native_);
    return pipeline;
}
bool LengthScalePipeline::HasDeviceV1() const noexcept { return bool(deviceV1Native_); }
std::shared_ptr<const ChargedBuffer> LengthScalePipeline::Candidate::keep() const noexcept {
    return succeeded() ? state_->keep : nullptr;
}
LengthScalePipeline::Candidate::~Candidate(){if(state_&&state_->pending)Quarantine();}
void LengthScalePipeline::Candidate::Quarantine() noexcept {if(!state_||!state_->pending||state_->lost)return;state_->lost=true;*state_->quarantine=state_;(void)state_->quarantine.release();}
std::shared_ptr<const ChargedBuffer> LengthScalePipeline::Candidate::output()const noexcept{return state_&&state_->proved&&state_->semantic==0?state_->output:nullptr;}
std::shared_ptr<const ChargedBuffer> LengthScalePipeline::Candidate::inputOwner()const noexcept{return state_?state_->input:nullptr;}
std::shared_ptr<const ChargedBuffer> LengthScalePipeline::Candidate::offsetsOwner()const noexcept{return state_?state_->offsets:nullptr;}
std::shared_ptr<const ChargedBuffer> LengthScalePipeline::Candidate::hairTOwner()const noexcept{return state_?state_->hairT:nullptr;}
std::shared_ptr<const ChargedBuffer> LengthScalePipeline::Candidate::stableIdsOwner()const noexcept{return state_?state_->stableIds:nullptr;}
uint32_t LengthScalePipeline::Candidate::count()const noexcept{return state_?state_->points:0;} uint32_t LengthScalePipeline::Candidate::curveCount()const noexcept{return state_?state_->curves:0;}
std::shared_ptr<DeviceContext> LengthScalePipeline::Candidate::context()const noexcept{return state_&&state_->native?state_->native->context:nullptr;}
bool LengthScalePipeline::Candidate::succeeded()const noexcept{return state_&&!state_->lost&&state_->proved&&state_->semantic==0;}
bool LengthScalePipeline::Candidate::usesReparam()const noexcept{return state_&&state_->native&&(state_->native->reparam||(state_->minimumFlavor&&state_->minimumControls.method==LengthScalePipeline::MinimumControls::Method::CutExtend&&state_->minimumControls.rebuild==LengthScalePipeline::MinimumControls::Rebuild::Reparam)||(state_->literalV1&&state_->literalV1Controls.method==LengthScalePipeline::LiteralV1Controls::Method::CutExtend&&state_->literalV1Controls.rebuild==LengthScalePipeline::LiteralV1Controls::Rebuild::Reparam)||(state_->deviceV1&&!state_->deviceV1Controls.cullOnly&&state_->deviceV1Controls.method==MinimumControls::Method::CutExtend&&state_->deviceV1Controls.rebuild==MinimumControls::Rebuild::Reparam)||(state_->envelopeV1&&!state_->envelopeV1Controls.cullOnly&&state_->envelopeV1Controls.method==LengthScalePipeline::EnvelopeV1Controls::Method::CutExtend&&state_->envelopeV1Controls.rebuild==LengthScalePipeline::EnvelopeV1Controls::Rebuild::Reparam));}
bool LengthScalePipeline::Candidate::usesHairT()const noexcept{return state_&&state_->native&&(state_->native->reparam||state_->minimumFlavor||state_->literalV1||state_->envelopeV1||state_->deviceV1);}
bool LengthScalePipeline::Candidate::usesStableIds()const noexcept{return state_&&(state_->literalV1||state_->envelopeV1||state_->deviceV1);}
VkResult LengthScalePipeline::Candidate::Poll(uint32_t* semanticStatus){if(!state_)return VK_ERROR_INITIALIZATION_FAILED;auto&s=*state_;if(s.lost)return VK_ERROR_DEVICE_LOST;if(!s.proved){VkResult r=vkGetFenceStatus(s.native->context->device(),s.fence);if(r==VK_NOT_READY)return r;if(r!=VK_SUCCESS){Quarantine();return r;}s.pending=false;void*data=nullptr;r=vkMapMemory(s.native->context->device(),s.status->memory(),0,4,0,&data);if(r!=VK_SUCCESS)return r;std::memcpy(&s.semantic,data,4);vkUnmapMemory(s.native->context->device(),s.status->memory());s.proved=true;}if(semanticStatus)*semanticStatus=s.semantic;return VK_SUCCESS;}
std::unique_ptr<LengthScalePipeline::Candidate> LengthScalePipeline::Begin(std::shared_ptr<const ChargedBuffer> points,std::shared_ptr<const ChargedBuffer> offsets,uint32_t curves,uint32_t pointCount,float factor,VkResult*result,BeforeSubmit beforeSubmit){
 return BeginImpl(native_,std::move(points),std::move(offsets),curves,pointCount,factor,result,std::move(beforeSubmit));
}
std::unique_ptr<LengthScalePipeline::Candidate> LengthScalePipeline::BeginSet(std::shared_ptr<const ChargedBuffer> points,std::shared_ptr<const ChargedBuffer> offsets,uint32_t curves,uint32_t pointCount,float target,VkResult*result,BeforeSubmit beforeSubmit){
 if (!setNative_) { if(result)*result=VK_ERROR_FEATURE_NOT_PRESENT; return {}; }
 return BeginImpl(setNative_,std::move(points),std::move(offsets),curves,pointCount,target,result,std::move(beforeSubmit));
}
std::unique_ptr<LengthScalePipeline::Candidate> LengthScalePipeline::BeginCutExtend(std::shared_ptr<const ChargedBuffer> points,std::shared_ptr<const ChargedBuffer> offsets,uint32_t curves,uint32_t pointCount,float value,bool absolute,VkResult*result,BeforeSubmit beforeSubmit){
 if (!cutNative_) { if(result)*result=VK_ERROR_FEATURE_NOT_PRESENT; return {}; }
 return BeginImpl(cutNative_,std::move(points),std::move(offsets),curves,pointCount,value,result,std::move(beforeSubmit),absolute ? 2u : 1u);
}
std::unique_ptr<LengthScalePipeline::Candidate> LengthScalePipeline::BeginReparam(std::shared_ptr<const ChargedBuffer> points,std::shared_ptr<const ChargedBuffer> offsets,std::shared_ptr<const ChargedBuffer> hairT,uint32_t curves,uint32_t pointCount,float value,bool absolute,VkResult*result,BeforeSubmit beforeSubmit){
 if (!reparamNative_) { if(result)*result=VK_ERROR_FEATURE_NOT_PRESENT; return {}; }
 return BeginImpl(reparamNative_,std::move(points),std::move(offsets),curves,pointCount,value,result,std::move(beforeSubmit),absolute ? 2u : 1u,std::move(hairT));
}
std::unique_ptr<LengthScalePipeline::Candidate> LengthScalePipeline::BeginMinimum(std::shared_ptr<const ChargedBuffer> points,std::shared_ptr<const ChargedBuffer> offsets,std::shared_ptr<const ChargedBuffer> hairT,uint32_t curves,uint32_t pointCount,MinimumControls const& controls,VkResult*result,BeforeSubmit beforeSubmit){
 if (!minimumNative_) { if(result)*result=VK_ERROR_FEATURE_NOT_PRESENT; return {}; }
 if (!std::isfinite(controls.value)||controls.value<0||!std::isfinite(controls.minimum)||controls.minimum<0||
     (controls.method!=MinimumControls::Method::Scale&&controls.method!=MinimumControls::Method::CutExtend)||
     (controls.rebuild!=MinimumControls::Rebuild::KeepParam&&controls.rebuild!=MinimumControls::Rebuild::Reparam)) { if(result)*result=VK_ERROR_INITIALIZATION_FAILED; return {}; }
 return BeginImpl(minimumNative_,std::move(points),std::move(offsets),curves,pointCount,controls.value,result,std::move(beforeSubmit),controls.absolute ? 2u : 1u,std::move(hairT),&controls);
}
std::unique_ptr<LengthScalePipeline::Candidate> LengthScalePipeline::BeginLiteralV1(std::shared_ptr<const ChargedBuffer> points,std::shared_ptr<const ChargedBuffer> offsets,std::shared_ptr<const ChargedBuffer> hairT,std::shared_ptr<const ChargedBuffer> stableIds,uint32_t curves,uint32_t pointCount,LiteralV1Controls const& controls,VkResult*result,BeforeSubmit beforeSubmit){
 if (!literalV1Native_) { if(result)*result=VK_ERROR_FEATURE_NOT_PRESENT; return {}; }
 if (!std::isfinite(controls.value)||controls.value<0||!std::isfinite(controls.minimum)||controls.minimum<0||!std::isfinite(controls.randomLo)||controls.randomLo<0||!std::isfinite(controls.randomHi)||controls.randomHi<0||
     (controls.method!=LiteralV1Controls::Method::Scale&&controls.method!=LiteralV1Controls::Method::CutExtend)||
     (controls.rebuild!=LiteralV1Controls::Rebuild::KeepParam&&controls.rebuild!=LiteralV1Controls::Rebuild::Reparam)) { if(result)*result=VK_ERROR_INITIALIZATION_FAILED; return {}; }
 return BeginImpl(literalV1Native_,std::move(points),std::move(offsets),curves,pointCount,controls.value,result,std::move(beforeSubmit),controls.absolute ? 2u : 1u,std::move(hairT),nullptr,std::move(stableIds),&controls);
}
std::unique_ptr<LengthScalePipeline::Candidate> LengthScalePipeline::BeginEnvelopeV1(std::shared_ptr<const ChargedBuffer> points,std::shared_ptr<const ChargedBuffer> offsets,std::shared_ptr<const ChargedBuffer> hairT,std::shared_ptr<const ChargedBuffer> stableIds,uint32_t curves,uint32_t pointCount,EnvelopeV1Controls const& controls,VkResult*result,BeforeSubmit beforeSubmit){
 if (!envelopeV1Native_) { if(result)*result=VK_ERROR_FEATURE_NOT_PRESENT; return {}; }
 if (!std::isfinite(controls.value)||controls.value<0||!std::isfinite(controls.minimum)||controls.minimum<0||!std::isfinite(controls.randomLo)||controls.randomLo<0||!std::isfinite(controls.randomHi)||controls.randomHi<0||!std::isfinite(controls.blend)||controls.blend<0||controls.blend>1||!std::isfinite(controls.maskAmount)||controls.maskAmount<0||controls.maskAmount>1||
     (controls.method!=EnvelopeV1Controls::Method::Scale&&controls.method!=EnvelopeV1Controls::Method::CutExtend)||
     (controls.rebuild!=EnvelopeV1Controls::Rebuild::KeepParam&&controls.rebuild!=EnvelopeV1Controls::Rebuild::Reparam)) { if(result)*result=VK_ERROR_INITIALIZATION_FAILED; return {}; }
 uint32_t const phase=controls.cullOnly?3u:(controls.absolute?2u:1u);
 return BeginImpl(envelopeV1Native_,std::move(points),std::move(offsets),curves,pointCount,controls.value,result,std::move(beforeSubmit),phase,std::move(hairT),nullptr,std::move(stableIds),nullptr,&controls);
}
std::unique_ptr<LengthScalePipeline::Candidate> LengthScalePipeline::BeginDeviceV1(
    std::shared_ptr<const ChargedBuffer> points, std::shared_ptr<const ChargedBuffer> offsets,
    std::shared_ptr<const ChargedBuffer> hairT, std::shared_ptr<const ChargedBuffer> stableIds,
    uint32_t curves, uint32_t pointCount, DeviceV1Controls const& controls,
    VkResult* result, BeforeSubmit beforeSubmit) {
    if (!deviceV1Native_) { if (result) *result = VK_ERROR_FEATURE_NOT_PRESENT; return {}; }
    auto nonnegative = [](float v) { return std::isfinite(v) && v >= 0; };
    if ((!controls.valueField.data && !nonnegative(controls.value)) ||
        (!controls.minimumField.data && !nonnegative(controls.minimum)) ||
        (!controls.randomField.data && (!nonnegative(controls.randomLo) || !nonnegative(controls.randomHi))) ||
        (!controls.cullThresholdField.data && !nonnegative(controls.cullThreshold)) ||
        !nonnegative(controls.blend) || controls.blend > 1 ||
        (!controls.maskField.data && (!nonnegative(controls.maskAmount) || controls.maskAmount > 1)) ||
        uint32_t(controls.method) > 1 || uint32_t(controls.rebuild) > 1) {
        if (result) *result = VK_ERROR_INITIALIZATION_FAILED; return {};
    }
    auto valid = [&](DeviceField const& f, uint32_t components, bool pointAllowed) {
        if (!f.data) return f.domain == 0;
        uint32_t count = f.domain == 1 ? 1u : f.domain == 2 ? curves : f.domain == 4 && pointAllowed ? pointCount : 0u;
        return (f.domain == 1 || f.domain == 2 || (f.domain == 4 && pointAllowed)) &&
            f.data->context() == context() && !f.data->unproven() && f.data->buffer() &&
            (f.data->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) &&
            f.data->sizeBytes() == std::max<VkDeviceSize>(4, VkDeviceSize(count) * components * 4);
    };
    if (!valid(controls.valueField, 1, true) || !valid(controls.minimumField, 1, true) ||
        !valid(controls.maskField, 1, true) || !valid(controls.randomField, 2, false) ||
        !valid(controls.cullThresholdField, 1, false) || !valid(controls.enabledField, 1, false)) {
        if (result) *result = VK_ERROR_INITIALIZATION_FAILED; return {};
    }
    uint32_t phase = controls.cullOnly ? 3u : (controls.absolute ? 2u : 1u);
    return BeginImpl(deviceV1Native_, std::move(points), std::move(offsets), curves, pointCount,
        controls.valueField.data ? 1.0f : controls.value, result, std::move(beforeSubmit), phase,
        std::move(hairT), nullptr, std::move(stableIds), nullptr, nullptr, &controls);
}
std::unique_ptr<LengthScalePipeline::Candidate> LengthScalePipeline::BeginImpl(std::shared_ptr<Native> selected,std::shared_ptr<const ChargedBuffer> points,std::shared_ptr<const ChargedBuffer> offsets,uint32_t curves,uint32_t pointCount,float factor,VkResult*result,BeforeSubmit beforeSubmit,uint32_t computePhase,std::shared_ptr<const ChargedBuffer> hairT,MinimumControls const* minimumControls,std::shared_ptr<const ChargedBuffer> stableIds,LiteralV1Controls const* literalV1Controls,EnvelopeV1Controls const* envelopeV1Controls,DeviceV1Controls const* deviceV1Controls){
 auto finish=[&](VkResult r){if(result)*result=r;};finish(VK_ERROR_INITIALIZATION_FAILED);auto context=selected->context;
 if (curves == UINT32_MAX || pointCount > UINT32_MAX / 3u) return {};
 VkDeviceSize pointBytes=VkDeviceSize(pointCount)*12,offsetBytes=VkDeviceSize(uint64_t(curves)+1)*4,hairTBytes=VkDeviceSize(pointCount)*4,stableIdBytes=VkDeviceSize(curves)*8;
 if(!std::isfinite(factor)||factor<0||(curves==0&&pointCount!=0)||(curves&&pointCount==0)||(pointCount&&!points)||(curves&&!points)||((curves||pointCount)&&!offsets)||
    (points&&(points->context()!=context||points->unproven()||!points->buffer()||!(points->usage()&VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)||points->sizeBytes()!=pointBytes))||
    (offsets&&(offsets->context()!=context||offsets->unproven()||!offsets->buffer()||!(offsets->usage()&VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)||offsets->sizeBytes()!=offsetBytes))||
    (hairT&&(!(selected->reparam||selected->minimumFlavor||selected->literalV1||selected->envelopeV1||selected->deviceV1)||hairT->context()!=context||hairT->unproven()||!hairT->buffer()||!(hairT->usage()&VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)||hairT->sizeBytes()!=hairTBytes))||
    (stableIds&&(!(selected->literalV1||selected->envelopeV1||selected->deviceV1)||stableIds->context()!=context||stableIds->unproven()||!stableIds->buffer()||!(stableIds->usage()&VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)||stableIds->sizeBytes()!=stableIdBytes)))return{};
 VkPhysicalDeviceProperties p{};vkGetPhysicalDeviceProperties(context->physicalDevice(),&p);
 // `offsets[curves]` is validated in phase zero. Widen before adding one,
 // so dispatch sizing cannot wrap before the curve-count guard above.
 uint64_t const offsetValidationCount=uint64_t(curves)+1u;
 uint64_t const work=std::max(offsetValidationCount,uint64_t(pointCount));
 uint64_t const groups=(work+255u)/256u;
 if (deviceV1Controls) {
     DeviceField const* fields[] = {&deviceV1Controls->valueField, &deviceV1Controls->randomField,
         &deviceV1Controls->minimumField, &deviceV1Controls->maskField,
         &deviceV1Controls->cullThresholdField, &deviceV1Controls->enabledField};
     for (auto f : fields) if (f->data && f->data->sizeBytes() > p.limits.maxStorageBufferRange) return {};
 }
 if(groups>p.limits.maxComputeWorkGroupCount[0]||pointBytes>p.limits.maxStorageBufferRange||offsetBytes>p.limits.maxStorageBufferRange||(hairT&&hairTBytes>p.limits.maxStorageBufferRange)||(stableIds&&stableIdBytes>p.limits.maxStorageBufferRange))return{};
 try{auto s=std::make_shared<Candidate::State>();s->native=selected;s->input=std::move(points);s->offsets=std::move(offsets);s->hairT=std::move(hairT);s->stableIds=std::move(stableIds);s->minimumFlavor=selected->minimumFlavor;s->literalV1=selected->literalV1;s->envelopeV1=selected->envelopeV1;s->deviceV1=selected->deviceV1;if(deviceV1Controls)s->deviceV1Controls=*deviceV1Controls;if(minimumControls)s->minimumControls=*minimumControls;if(literalV1Controls)s->literalV1Controls=*literalV1Controls;if(envelopeV1Controls)s->envelopeV1Controls=*envelopeV1Controls;s->points=pointCount;s->curves=curves;s->quarantine=std::make_unique<std::shared_ptr<Candidate::State>>();auto candidate=std::unique_ptr<Candidate>(new Candidate(s));if(!pointCount&&!curves){s->proved=true;s->semantic=0;finish(VK_SUCCESS);return candidate;}
  VkBufferCreateInfo bi{};bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;bi.size=pointBytes;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;VkResult r;s->output=ChargedBuffer::Create(context,bi,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,UsdGenExecutionResourceKind::Active,&r);if(!s->output){finish(r);return{};}bi.size=4;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;s->status=ChargedBuffer::Create(context,bi,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,UsdGenExecutionResourceKind::Scratch,&r);if(!s->status){finish(r);return{};}auto d=context->device();void*data=nullptr;r=vkMapMemory(d,s->status->memory(),0,4,0,&data);if(r!=VK_SUCCESS){finish(r);return{};}std::memset(data,0,4);vkUnmapMemory(d,s->status->memory());
  if(selected->deviceV1){bi.size=VkDeviceSize(curves)*4;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT;s->keep=ChargedBuffer::Create(context,bi,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,UsdGenExecutionResourceKind::Scratch,&r);if(!s->keep){finish(r);return{};}}
  uint32_t const descriptorCount=selected->deviceV1?13u:(selected->literalV1||selected->envelopeV1||selected->deviceV1)?6u:((selected->reparam||selected->minimumFlavor)?5u:4u);VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,descriptorCount};VkDescriptorPoolCreateInfo dp{};dp.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;dp.maxSets=1;dp.poolSizeCount=1;dp.pPoolSizes=&ps;r=vkCreateDescriptorPool(d,&dp,nullptr,&s->descriptors);if(r!=VK_SUCCESS){finish(r);return{};}VkDescriptorSetAllocateInfo da{};da.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;da.descriptorPool=s->descriptors;da.descriptorSetCount=1;da.pSetLayouts=&selected->descriptors;VkDescriptorSet set;r=vkAllocateDescriptorSets(d,&da,&set);if(r!=VK_SUCCESS){finish(r);return{};}VkDescriptorBufferInfo infos[13]={{s->input->buffer(),0,pointBytes},{s->offsets->buffer(),0,offsetBytes},{s->output->buffer(),0,pointBytes},{s->status->buffer(),0,4},{s->hairT?s->hairT->buffer():s->offsets->buffer(),0,s->hairT?hairTBytes:offsetBytes},{s->stableIds?s->stableIds->buffer():s->offsets->buffer(),0,s->stableIds?stableIdBytes:offsetBytes}};if(selected->deviceV1){DeviceField const* fields[]={&s->deviceV1Controls.valueField,&s->deviceV1Controls.randomField,&s->deviceV1Controls.minimumField,&s->deviceV1Controls.maskField,&s->deviceV1Controls.cullThresholdField,&s->deviceV1Controls.enabledField};for(uint32_t i=0;i<6;++i){auto owner=fields[i]->data?fields[i]->data:s->offsets;infos[6+i]={owner->buffer(),0,owner->sizeBytes()};}infos[12]={s->keep->buffer(),0,s->keep->sizeBytes()};}VkWriteDescriptorSet writes[13]{};for(uint32_t i=0;i<descriptorCount;++i){writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];}vkUpdateDescriptorSets(d,descriptorCount,writes,0,nullptr);
  VkCommandPoolCreateInfo pc{};pc.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;pc.queueFamilyIndex=context->computeQueueFamily();r=vkCreateCommandPool(d,&pc,nullptr,&s->commands);if(r!=VK_SUCCESS){finish(r);return{};}VkCommandBufferAllocateInfo ca{};ca.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;ca.commandPool=s->commands;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;VkCommandBuffer cmd;r=vkAllocateCommandBuffers(d,&ca,&cmd);if(r!=VK_SUCCESS){finish(r);return{};}VkCommandBufferBeginInfo cb{};cb.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;r=vkBeginCommandBuffer(cmd,&cb);if(r!=VK_SUCCESS){finish(r);return{};}
  VkMemoryBarrier before{};before.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;before.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT;before.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,nullptr,0,nullptr);vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,selected->pipeline);vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,selected->layout,0,1,&set,0,nullptr);struct Controls{uint32_t curves,points,phase;float factor;};struct ReparamControls{uint32_t curves,points,phase;float value;uint32_t hasHairT;};struct MinimumPush{uint32_t curves,points,phase;float value;uint32_t hasHairT;float minimum;uint32_t method,rebuild;};struct LiteralV1Push{uint32_t curves,points,phase;float value;uint32_t hasHairT;float minimum;uint32_t method,rebuild;float randomLo,randomHi;uint32_t seedBits,hasStableIds;};struct EnvelopeV1Push{uint32_t curves,points,phase;float value;uint32_t hasHairT;float minimum;uint32_t method,rebuild;float randomLo,randomHi;uint32_t seedBits,hasStableIds;float resolvedEnvelope;uint32_t zeroEnvelope;};struct DeviceV1Push{uint32_t curves,points,phase;float value;uint32_t hasHairT;float minimum;uint32_t method,rebuild;float randomLo,randomHi;uint32_t seedBits,hasStableIds;float blend,maskAmount,threshold;uint32_t enabled,valueDom,randomDom,minimumDom,maskDom,thresholdDom,enabledDom;};static_assert(sizeof(DeviceV1Push)==88,"device v1 shader ABI");static_assert(sizeof(Controls)==16,"length shader ABI");static_assert(sizeof(ReparamControls)==20,"reparam shader ABI");static_assert(sizeof(MinimumPush)==32,"minimum shader ABI");static_assert(sizeof(LiteralV1Push)==48,"literal v1 shader ABI");static_assert(sizeof(EnvelopeV1Push)==56,"envelope v1 shader ABI");Controls c{curves,pointCount,0,factor};ReparamControls rc{curves,pointCount,0,factor,s->hairT?1u:0u};MinimumPush mc{curves,pointCount,0,factor,s->hairT?1u:0u,s->minimumControls.minimum,uint32_t(s->minimumControls.method),uint32_t(s->minimumControls.rebuild)};uint32_t seedBits;std::memcpy(&seedBits,&s->literalV1Controls.seed,4);LiteralV1Push lc{curves,pointCount,0,factor,s->hairT?1u:0u,s->literalV1Controls.minimum,uint32_t(s->literalV1Controls.method),uint32_t(s->literalV1Controls.rebuild),s->literalV1Controls.randomLo,s->literalV1Controls.randomHi,seedBits,s->stableIds?1u:0u};uint32_t envelopeSeedBits;std::memcpy(&envelopeSeedBits,&s->envelopeV1Controls.seed,4);float const resolvedEnvelope=ResolveLengthEnvelope(s->envelopeV1Controls.blend,s->envelopeV1Controls.maskAmount);EnvelopeV1Push ec{curves,pointCount,0,factor,s->hairT?1u:0u,s->envelopeV1Controls.minimum,uint32_t(s->envelopeV1Controls.method),uint32_t(s->envelopeV1Controls.rebuild),s->envelopeV1Controls.randomLo,s->envelopeV1Controls.randomHi,envelopeSeedBits,s->stableIds?1u:0u,resolvedEnvelope,resolvedEnvelope==0?1u:0u};auto const& dc=s->deviceV1Controls;uint32_t ds;std::memcpy(&ds,&dc.seed,4);DeviceV1Push dpc{curves,pointCount,0,dc.value,s->hairT?1u:0u,dc.minimum,uint32_t(dc.method),uint32_t(dc.rebuild),dc.randomLo,dc.randomHi,ds,s->stableIds?1u:0u,dc.blend,dc.maskAmount,dc.cullThreshold,dc.enabled?1u:0u,dc.valueField.domain,dc.randomField.domain,dc.minimumField.domain,dc.maskField.domain,dc.cullThresholdField.domain,dc.enabledField.domain};if(selected->deviceV1)vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,88,&dpc);else if(selected->envelopeV1)vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,56,&ec);else if(selected->literalV1)vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,48,&lc);else if(selected->minimumFlavor)vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,32,&mc);else if(selected->reparam)vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,20,&rc);else vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,&c);vkCmdDispatch(cmd,uint32_t(groups),1,1);
  VkMemoryBarrier gate{};gate.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;gate.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;gate.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&gate,0,nullptr,0,nullptr);c.phase=computePhase;rc.phase=computePhase;mc.phase=computePhase;lc.phase=computePhase;ec.phase=computePhase;dpc.phase=computePhase;if(selected->deviceV1)vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,88,&dpc);else if(selected->envelopeV1)vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,56,&ec);else if(selected->literalV1)vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,48,&lc);else if(selected->minimumFlavor)vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,32,&mc);else if(selected->reparam)vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,20,&rc);else vkCmdPushConstants(cmd,selected->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,16,&c);vkCmdDispatch(cmd,uint32_t(groups),1,1);VkMemoryBarrier after{};after.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;after.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;after.dstAccessMask=VK_ACCESS_HOST_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_SHADER_READ_BIT;vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&after,0,nullptr,0,nullptr);r=vkEndCommandBuffer(cmd);if(r!=VK_SUCCESS){finish(r);return{};}VkFenceCreateInfo fi{};fi.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;r=vkCreateFence(d,&fi,nullptr,&s->fence);if(r!=VK_SUCCESS){finish(r);return{};}if(beforeSubmit){bool admitted=false;try{admitted=beforeSubmit();}catch(...){finish(VK_ERROR_UNKNOWN);return{};}if(!admitted){finish(VK_ERROR_OUT_OF_DEVICE_MEMORY);return{};}}VkSubmitInfo si{};si.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;si.commandBufferCount=1;si.pCommandBuffers=&cmd;s->pending=true;r=vkQueueSubmit(context->computeQueue(),1,&si,s->fence);if(r!=VK_SUCCESS){candidate->Quarantine();finish(r);return{};}finish(VK_SUCCESS);return candidate;
 }catch(std::bad_alloc const&){finish(VK_ERROR_OUT_OF_HOST_MEMORY);return{};}
}
} // namespace usdGen::vulkan
