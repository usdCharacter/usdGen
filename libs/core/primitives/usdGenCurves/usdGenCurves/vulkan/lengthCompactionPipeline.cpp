// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "usdGenCurves/vulkan/lengthCompactionPipeline.h"
#include "usdGenCurves/vulkan/sourceGeneration.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace usdGen::vulkan {
namespace {
struct Push {
    uint32_t curves=0, points=0, phase=0, distance=0;
    float threshold=0;
    uint32_t outputWords=0, stride=0, inputBytes=0, outputBytes=0, domain=0, spanCount=0;
};
static_assert(sizeof(Push)==44, "Length compaction shader ABI");
void Barrier(VkCommandBuffer cmd) {
    VkMemoryBarrier b{}; b.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT;
    b.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_READ_BIT|
        VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT|VK_PIPELINE_STAGE_HOST_BIT,
        VK_PIPELINE_STAGE_ALL_COMMANDS_BIT|VK_PIPELINE_STAGE_HOST_BIT,0,1,&b,0,nullptr,0,nullptr);
}
uint64_t RoundWord(uint64_t bytes) { return (bytes+3u)&~uint64_t(3u); }
}
struct LengthCompactionPipeline::Native {
    bool supportsKeep = false;
    std::shared_ptr<DeviceContext> context;
    VkShaderModule shader=VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors=VK_NULL_HANDLE;
    VkPipelineLayout layout=VK_NULL_HANDLE;
    VkPipeline pipeline=VK_NULL_HANDLE;
    VkPhysicalDeviceLimits limits{};
    ~Native() {
        auto d=context->device();
        if(pipeline) vkDestroyPipeline(d,pipeline,nullptr);
        if(layout) vkDestroyPipelineLayout(d,layout,nullptr);
        if(descriptors) vkDestroyDescriptorSetLayout(d,descriptors,nullptr);
        if(shader) vkDestroyShaderModule(d,shader,nullptr);
    }
};
struct LengthCompactionPipeline::Candidate::State : std::enable_shared_from_this<State> {
    struct Phase { VkDescriptorPool descriptors=VK_NULL_HANDLE; VkCommandPool commands=VK_NULL_HANDLE;
        VkCommandBuffer command=VK_NULL_HANDLE; uint64_t value=0; };  // value: queue timeline
    std::shared_ptr<Native> native;
    std::shared_ptr<const VulkanSourceGeneration> base;
    std::vector<std::shared_ptr<ChargedBuffer>> owned;
    std::shared_ptr<ChargedBuffer> dummy,status,spans,keep,scanA,scanB,reverseCurve,reversePoint;
    std::shared_ptr<const ChargedBuffer> finalScan, inputKeep;
    std::vector<OutputPlane> outputs;
    std::vector<UsdGenChunkDesc> chunks;
    std::vector<UsdGenDeviceTileMetadata> tiles;
    std::vector<uint32_t> control;
    Phase phases[2];
    Push push;
    uint32_t curves=0,points=0,semantic=UINT32_MAX;
    bool pending=false,lost=false,countsProved=false,scatterStarted=false,scatterSubmitted=false,scatterProved=false;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() {
        auto d=native->context->device();
        for(auto& p:phases) {
            if(p.commands) vkDestroyCommandPool(d,p.commands,nullptr);
            if(p.descriptors) vkDestroyDescriptorPool(d,p.descriptors,nullptr);
        }
    }
    void Retain() noexcept {
        if(!pending||lost) return;
        lost=true; *quarantine=shared_from_this(); (void)quarantine.release();
    }
    std::shared_ptr<ChargedBuffer> Buffer(uint64_t bytes,bool host,VkResult& r,
        UsdGenExecutionResourceKind kind=UsdGenExecutionResourceKind::Scratch) {
        bytes=std::max<uint64_t>(4,bytes);
        if(bytes>UINT32_MAX||bytes>native->limits.maxStorageBufferRange) {
            r=VK_ERROR_FEATURE_NOT_PRESENT; return {};
        }
        VkBufferCreateInfo b{}; b.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO; b.size=bytes;
        b.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        b.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
        auto out=ChargedBuffer::Create(native->context,b,host ?
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT:
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,kind,&r);
        if(out) owned.push_back(out);
        return out;
    }
    bool Write(std::shared_ptr<ChargedBuffer> const& b,void const* data,size_t bytes,VkResult& r) {
        void* target=nullptr;
        r=vkMapMemory(native->context->device(),b->memory(),0,b->sizeBytes(),0,&target);
        if(r!=VK_SUCCESS) return false;
        std::memset(target,0,size_t(b->sizeBytes()));
        if(bytes) std::memcpy(target,data,bytes);
        vkUnmapMemory(native->context->device(),b->memory()); return true;
    }
    bool Setup(unsigned which,uint32_t sets,VkResult& r) {
        auto& p=phases[which]; auto d=native->context->device();
        VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,sets*10u};
        VkDescriptorPoolCreateInfo dp{}; dp.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dp.maxSets=sets;dp.poolSizeCount=1;dp.pPoolSizes=&size;
        r=vkCreateDescriptorPool(d,&dp,nullptr,&p.descriptors);if(r!=VK_SUCCESS)return false;
        VkCommandPoolCreateInfo cp{}; cp.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cp.queueFamilyIndex=native->context->computeQueueFamily();
        r=vkCreateCommandPool(d,&cp,nullptr,&p.commands);if(r!=VK_SUCCESS)return false;
        VkCommandBufferAllocateInfo ca{}; ca.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        ca.commandPool=p.commands;ca.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ca.commandBufferCount=1;
        r=vkAllocateCommandBuffers(d,&ca,&p.command);if(r!=VK_SUCCESS)return false;
        VkCommandBufferBeginInfo begin{}; begin.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        r=vkBeginCommandBuffer(p.command,&begin);if(r!=VK_SUCCESS)return false;
        Barrier(p.command); vkCmdBindPipeline(p.command,VK_PIPELINE_BIND_POINT_COMPUTE,native->pipeline);
        return true;
    }
    using Bindings=std::array<std::shared_ptr<const ChargedBuffer>,10>;
    Bindings Bind() const {
        Bindings b; b.fill(dummy);
        b[0]=base->PlaneOwner(UsdGenCurveChannels().points); if(!b[0])b[0]=dummy;
        b[1]=base->PlaneOwner(UsdGenCurveChannels().curveOffsets); if(!b[1])b[1]=dummy;
        b[2]=keep;b[3]=finalScan?finalScan:scanA;b[4]=scanB;b[5]=status;b[6]=spans;
        if(reverseCurve)b[7]=reverseCurve;
        if(reversePoint)b[8]=reversePoint;
        return b;
    }
    bool Dispatch(unsigned which,Bindings const& bindings,Push const& controls,uint64_t work,VkResult& r) {
        uint64_t groups=(std::max<uint64_t>(work,1)+63u)/64u;
        if(groups>native->limits.maxComputeWorkGroupCount[0]) {r=VK_ERROR_FEATURE_NOT_PRESENT;return false;}
        auto& p=phases[which];auto d=native->context->device();
        VkDescriptorSetAllocateInfo a{}; a.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        a.descriptorPool=p.descriptors;a.descriptorSetCount=1;a.pSetLayouts=&native->descriptors;
        VkDescriptorSet set=VK_NULL_HANDLE;r=vkAllocateDescriptorSets(d,&a,&set);if(r!=VK_SUCCESS)return false;
        VkDescriptorBufferInfo infos[10]{};VkWriteDescriptorSet writes[10]{};
        for(uint32_t i=0;i<10;++i) {
            infos[i]={bindings[i]->buffer(),0,bindings[i]->sizeBytes()};
            writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set;writes[i].dstBinding=i;
            writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[i].pBufferInfo=&infos[i];
        }
        vkUpdateDescriptorSets(d,10,writes,0,nullptr);
        vkCmdBindDescriptorSets(p.command,VK_PIPELINE_BIND_POINT_COMPUTE,native->layout,0,1,&set,0,nullptr);
        vkCmdPushConstants(p.command,native->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(Push),&controls);
        vkCmdDispatch(p.command,uint32_t(groups),1,1);Barrier(p.command);return true;
    }
    bool Submit(unsigned which,BeforeSubmit const& before,VkResult& r) {
        auto& p=phases[which];Barrier(p.command);
        r=vkEndCommandBuffer(p.command);if(r!=VK_SUCCESS)return false;
        if(before) {
            bool admitted=false;
            try {admitted=before();}catch(...){r=VK_ERROR_UNKNOWN;return false;}
            if(!admitted){r=VK_ERROR_OUT_OF_DEVICE_MEMORY;return false;}
        }
        pending=true;p.value=native->context->computeTimeline()->Submit(&p.command,1,&r);
        if(!p.value){Retain();return false;}return true;
    }
    VkResult Prove(unsigned which) {
        if(lost)return VK_ERROR_DEVICE_LOST;
        if(!phases[which].value)return VK_NOT_READY;  // not submitted
        VkResult r=native->context->computeTimeline()->Check(phases[which].value);
        if(r==VK_NOT_READY)return r;
        if(r!=VK_SUCCESS){Retain();return r;}
        pending=false;void* data=nullptr;
        r=vkMapMemory(native->context->device(),status->memory(),0,status->sizeBytes(),0,&data);
        if(r!=VK_SUCCESS)return r;
        std::memcpy(control.data(),data,control.size()*sizeof(uint32_t));
        vkUnmapMemory(native->context->device(),status->memory());semantic=control[0];return VK_SUCCESS;
    }
};

LengthCompactionPipeline::LengthCompactionPipeline(std::shared_ptr<Native> n):native_(std::move(n)){}
LengthCompactionPipeline::~LengthCompactionPipeline()=default;
std::shared_ptr<DeviceContext> const& LengthCompactionPipeline::context()const noexcept{return native_->context;}
std::shared_ptr<LengthCompactionPipeline> LengthCompactionPipeline::Create(
    std::shared_ptr<DeviceContext> context,std::vector<uint32_t> const& code,VkResult* result) {
    VkResult r=VK_ERROR_INITIALIZATION_FAILED; if(result)*result=r;
    if(!context||code.size()<5||code[0]!=0x07230203u)return {};
    // Submissions go through the queue timeline.
    if(!context->computeTimeline()){if(result)*result=VK_ERROR_FEATURE_NOT_PRESENT;return {};}
    try {
        auto n=std::make_shared<Native>();n->context=std::move(context);
        VkPhysicalDeviceProperties properties{};vkGetPhysicalDeviceProperties(n->context->physicalDevice(),&properties);
        n->limits=properties.limits;
        if(n->limits.maxComputeWorkGroupInvocations<64||n->limits.maxComputeWorkGroupSize[0]<64||
            n->limits.maxPerStageDescriptorStorageBuffers<10||n->limits.maxDescriptorSetStorageBuffers<10)return {};
        auto d=n->context->device();VkShaderModuleCreateInfo sm{};sm.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        sm.codeSize=code.size()*4;sm.pCode=code.data();r=vkCreateShaderModule(d,&sm,nullptr,&n->shader);
        if(r!=VK_SUCCESS){if(result)*result=r;return {};}
        VkDescriptorSetLayoutBinding bindings[10]{};
        for(uint32_t i=0;i<10;++i)bindings[i]={i,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
        VkDescriptorSetLayoutCreateInfo ds{};ds.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;ds.bindingCount=10;ds.pBindings=bindings;
        r=vkCreateDescriptorSetLayout(d,&ds,nullptr,&n->descriptors);if(r!=VK_SUCCESS){if(result)*result=r;return {};}
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(Push)};
        VkPipelineLayoutCreateInfo pl{};pl.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;pl.setLayoutCount=1;pl.pSetLayouts=&n->descriptors;
        pl.pushConstantRangeCount=1;pl.pPushConstantRanges=&push;
        r=vkCreatePipelineLayout(d,&pl,nullptr,&n->layout);if(r!=VK_SUCCESS){if(result)*result=r;return {};}
        VkComputePipelineCreateInfo cp{};cp.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;cp.layout=n->layout;
        cp.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
        cp.stage.module=n->shader;cp.stage.pName="main";
        r=vkCreateComputePipelines(d,VK_NULL_HANDLE,1,&cp,nullptr,&n->pipeline);if(r!=VK_SUCCESS){if(result)*result=r;return {};}
        auto out=std::shared_ptr<LengthCompactionPipeline>(new LengthCompactionPipeline(std::move(n)));
        if(result)*result=VK_SUCCESS;
        return out;
    }catch(std::bad_alloc const&){if(result)*result=VK_ERROR_OUT_OF_HOST_MEMORY;return {};}
}
LengthCompactionPipeline::Candidate::Candidate(std::shared_ptr<State> s):state_(std::move(s)){}
LengthCompactionPipeline::Candidate::~Candidate(){if(state_&&state_->pending)Quarantine();}
void LengthCompactionPipeline::Candidate::Quarantine()noexcept{if(state_)state_->Retain();}
std::shared_ptr<const VulkanSourceGeneration> LengthCompactionPipeline::Candidate::inputOwner()const noexcept{return state_->base;}
std::shared_ptr<DeviceContext> LengthCompactionPipeline::Candidate::context()const noexcept{return state_->native->context;}
uint32_t LengthCompactionPipeline::Candidate::curveCount()const noexcept{return state_->curves;}
uint32_t LengthCompactionPipeline::Candidate::pointCount()const noexcept{return state_->points;}
bool LengthCompactionPipeline::Candidate::succeeded()const noexcept{return state_->scatterProved&&!state_->lost&&!state_->semantic;}
namespace { std::vector<LengthCompactionPipeline::Candidate::OutputPlane> const kNoCompactionOutputs; }
std::vector<LengthCompactionPipeline::Candidate::OutputPlane> const& LengthCompactionPipeline::Candidate::outputs()const noexcept{
    return succeeded()?state_->outputs:kNoCompactionOutputs;
}
std::vector<UsdGenChunkDesc> const& LengthCompactionPipeline::Candidate::chunks()const noexcept{return state_->chunks;}
std::vector<UsdGenDeviceTileMetadata> const& LengthCompactionPipeline::Candidate::tiles()const noexcept{return state_->tiles;}

std::shared_ptr<LengthCompactionPipeline> LengthCompactionPipeline::CreateWithKeep(
    std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code, VkResult* result) {
    auto pipeline = Create(std::move(context), code, result);
    if (pipeline) pipeline->native_->supportsKeep = true;
    return pipeline;
}
bool LengthCompactionPipeline::HasKeep() const noexcept { return native_->supportsKeep; }
std::unique_ptr<LengthCompactionPipeline::Candidate> LengthCompactionPipeline::Begin(
    std::shared_ptr<const VulkanSourceGeneration> base,float threshold,VkResult* result,BeforeSubmit before) {
    return BeginImpl(std::move(base), threshold, {}, false, result, std::move(before));
}
std::unique_ptr<LengthCompactionPipeline::Candidate> LengthCompactionPipeline::BeginWithKeep(
    std::shared_ptr<const VulkanSourceGeneration> base, std::shared_ptr<const ChargedBuffer> keep,
    VkResult* result, BeforeSubmit before) {
    if (!HasKeep()) { if (result) *result = VK_ERROR_FEATURE_NOT_PRESENT; return {}; }
    return BeginImpl(std::move(base), 0, std::move(keep), true, result, std::move(before));
}
std::unique_ptr<LengthCompactionPipeline::Candidate> LengthCompactionPipeline::BeginImpl(
    std::shared_ptr<const VulkanSourceGeneration> base, float threshold,
    std::shared_ptr<const ChargedBuffer> keep, bool useKeep, VkResult* result, BeforeSubmit before) {
    VkResult r=VK_ERROR_INITIALIZATION_FAILED;if(result)*result=r;
    if (useKeep && base && base->curveCount() && (!keep || keep->context() != context() ||
        keep->unproven() || !keep->buffer() || !(keep->usage() & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) ||
        keep->sizeBytes() != VkDeviceSize(base->curveCount()) * 4 ||
        keep->sizeBytes() > native_->limits.maxStorageBufferRange)) return {};
    if(!base||base->context()!=context()||!std::isfinite(threshold)||threshold<0||
        uint64_t(base->pointCount())*12>UINT32_MAX||uint64_t(base->curveCount())*8+8>UINT32_MAX)return {};
    try {
        auto s=std::make_shared<Candidate::State>();s->native=native_;s->base=std::move(base);s->inputKeep=std::move(keep);
        s->quarantine=std::make_unique<std::shared_ptr<Candidate::State>>();
        auto candidate=std::unique_ptr<Candidate>(new Candidate(s));
        s->push.curves=s->base->curveCount();s->push.points=s->base->pointCount();s->push.threshold=threshold;
        s->chunks=s->base->chunks();s->tiles=s->base->geometry().tiles;
        size_t spanCount=s->chunks.size()*2+s->tiles.size();
        if(spanCount>(UINT32_MAX/4-3)/4)return {};
        s->push.spanCount=uint32_t(spanCount);s->control.assign(3+spanCount*4,0);
        std::vector<uint32_t> spans(spanCount*4,0);
        size_t index=0;
        // Chunk point ends may be ragged; express them as a sentinel to be
        // resolved by the shader from the immutable offsets, never host data.
        for(auto const& c:s->chunks) {
            uint64_t end=uint64_t(c.firstCurve)+c.curveCount;
            if(end>s->push.curves||c.liveCount>c.curveCount||c.firstCv>s->push.points)return {};
            spans[index++]=c.firstCurve;spans[index++]=uint32_t(end);
            spans[index++]=c.firstCv;spans[index++]=UINT32_MAX;
            spans[index++]=c.firstCurve;spans[index++]=c.firstCurve+c.liveCount;
            spans[index++]=c.firstCv;spans[index++]=UINT32_MAX;
        }
        for(auto const& t:s->tiles) {
            if(t.firstElement>s->push.curves||t.elementCount>s->push.curves-t.firstElement||
                t.firstPoint>s->push.points||t.pointCount>s->push.points-t.firstPoint)return {};
            spans[index++]=uint32_t(t.firstElement);spans[index++]=uint32_t(t.firstElement+t.elementCount);
            spans[index++]=uint32_t(t.firstPoint);spans[index++]=uint32_t(t.firstPoint+t.pointCount);
        }
        auto valid=[&](std::shared_ptr<const ChargedBuffer> const& b,uint64_t bytes) {
            return !bytes ? !b || (b->context()==context()&&!b->unproven()) : b&&b->context()==context()&&
                !b->unproven()&&b->sizeBytes()>=bytes&&b->sizeBytes()<=native_->limits.maxStorageBufferRange&&
                (b->usage()&VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        };
        if(!valid(s->base->PlaneOwner(UsdGenCurveChannels().points),uint64_t(s->push.points)*12)||
            !valid(s->base->PlaneOwner(UsdGenCurveChannels().curveOffsets),(uint64_t(s->push.curves)+1)*4))return {};
        s->dummy=s->Buffer(4,true,r);if(!s->dummy||!s->Write(s->dummy,nullptr,0,r))goto failed;
        s->status=s->Buffer(s->control.size()*4,true,r);
        s->spans=s->Buffer(spans.size()*4,true,r);
        s->keep=s->Buffer(uint64_t(s->push.curves)*4,false,r);
        s->scanA=s->Buffer(uint64_t(s->push.curves)*8,false,r);
        s->scanB=s->Buffer(uint64_t(s->push.curves)*8,false,r);
        if(!s->status||!s->spans||!s->keep||!s->scanA||!s->scanB||
            !s->Write(s->status,nullptr,0,r)||!s->Write(s->spans,spans.data(),spans.size()*4,r))goto failed;
        if(!s->Setup(0,35,r))goto failed;
        {
            auto b=s->Bind();b[4]=s->scanA;
            auto countPush=s->push;countPush.distance=useKeep?1u:0u;if(s->inputKeep)b[3]=s->inputKeep;
            if(!s->Dispatch(0,b,countPush,s->push.curves,r))goto failed;
            auto from=s->scanA,to=s->scanB;
            for(uint64_t distance=1;distance<s->push.curves;distance*=2) {
                auto p=s->push;p.phase=1;p.distance=uint32_t(distance);b[3]=from;b[4]=to;
                if(!s->Dispatch(0,b,p,s->push.curves,r))goto failed;
                std::swap(from,to);
            }
            s->finalScan=from;b=s->Bind();auto p=s->push;p.phase=2;
            if(!s->Dispatch(0,b,p,spanCount,r)||!s->Submit(0,before,r))goto failed;
        }
        if(result)*result=VK_SUCCESS;
        return candidate;
    failed:
        if(result)*result=r;
        return {};
    }catch(std::bad_alloc const&){if(result)*result=VK_ERROR_OUT_OF_HOST_MEMORY;return {};}
}
VkResult LengthCompactionPipeline::Candidate::PollCounts(uint32_t* semanticStatus) {
    auto& s=*state_;if(s.lost)return VK_ERROR_DEVICE_LOST;
    if(!s.countsProved) {
        auto r=s.Prove(0);if(r!=VK_SUCCESS)return r;
        s.curves=s.control[1];s.points=s.control[2];
        if(s.curves>s.push.curves||s.points>s.push.points)s.semantic|=1;
        size_t i=3;
        for(auto& c:s.chunks) {
            c.firstCurve=s.control[i++];c.curveCount=s.control[i++];
            c.firstCv=s.control[i++];++i; // CV arity is unchanged by pure filtering.
            ++i;c.liveCount=s.control[i++];i+=2;
        }
        for(auto& t:s.tiles) {
            t.firstElement=s.control[i++];t.elementCount=s.control[i++];
            t.firstPoint=s.control[i++];t.pointCount=s.control[i++];t.boundsValid=false;
        }
        s.countsProved=true;
    }
    if(semanticStatus)*semanticStatus=s.semantic;
    return VK_SUCCESS;
}
bool LengthCompactionPipeline::Candidate::BeginScatter(VkResult* result,BeforeSubmit before) {
    auto& s=*state_;VkResult r=VK_ERROR_INITIALIZATION_FAILED;if(result)*result=r;
    if(!s.countsProved||s.semantic||s.lost||s.scatterStarted)return false;
    s.scatterStarted=true;
    try {
        auto total=s.base->planes().size()+s.base->sourceFrames().size();
        if(total>(UINT32_MAX/10)-2)return false;
        s.outputs.reserve(total);
        s.reverseCurve=s.Buffer(uint64_t(s.curves)*4,false,r);
        s.reversePoint=s.Buffer(uint64_t(s.points)*4,false,r);
        if(!s.reverseCurve||!s.reversePoint||!s.Setup(1,uint32_t(total+2),r))goto failed;
        {
            auto b=s.Bind();auto p=s.push;p.phase=3;
            if(!s.Dispatch(1,b,p,s.push.curves,r))goto failed;
            auto gather=[&](VulkanSourceGeneration::PlaneView const& plane,bool privateFrame)->bool {
                OutputPlane output{plane.metadata,{},privateFrame,0};
                auto owner=privateFrame?s.base->SourceFrameOwner(plane.metadata.name):s.base->PlaneOwner(plane.metadata.name);
                if(plane.bytes&&(!owner||owner->context()!=context()||owner->unproven()||
                    owner->sizeBytes()<plane.bytes||!(owner->usage()&VK_BUFFER_USAGE_TRANSFER_SRC_BIT)))return false;
                if(plane.metadata.domain==UsdGenDeviceDomain::Groom) {
                    output.owner=owner;output.bytes=plane.bytes;s.outputs.push_back(std::move(output));return true;
                }
                bool offsets=plane.metadata.semantic==UsdGenDeviceChannelSemantic::ElementOffsets;
                bool point=plane.metadata.domain==UsdGenDeviceDomain::Point;
                if(!offsets&&!point&&plane.metadata.domain!=UsdGenDeviceDomain::Primitive)return false;
                output.metadata.elementCount=offsets?uint64_t(s.curves)+1:point?s.points:s.curves;
                size_t bytes=0;
                if(!UsdGenDeviceChannelStorageBytes(output.metadata,&bytes,nullptr)||bytes>UINT32_MAX||
                    plane.bytes>UINT32_MAX)return false;
                output.bytes=bytes;
                if(!bytes){s.outputs.push_back(std::move(output));return true;}
                auto out=s.Buffer(RoundWord(bytes),false,r,UsdGenExecutionResourceKind::Active);if(!out)return false;
                output.owner=out;b=s.Bind();b[9]=out;p=s.push;p.phase=offsets?5:4;
                p.outputBytes=uint32_t(bytes);p.outputWords=uint32_t(RoundWord(bytes)/4);
                if(!offsets) {
                    auto packed=s.Buffer(RoundWord(plane.bytes),false,r);if(!packed)return false;
                    vkCmdFillBuffer(s.phases[1].command,packed->buffer(),0,packed->sizeBytes(),0);Barrier(s.phases[1].command);
                    if(plane.bytes) {
                        VkBufferCopy copy{0,0,plane.bytes};
                        vkCmdCopyBuffer(s.phases[1].command,owner->buffer(),packed->buffer(),1,&copy);
                        Barrier(s.phases[1].command);
                    }
                    b[0]=packed;p.inputBytes=uint32_t(plane.bytes);p.domain=point?0:1;
                    auto one=plane.metadata;one.elementCount=1;size_t elementBytes=0;
                    if(!UsdGenDeviceChannelStorageBytes(one,&elementBytes,nullptr)||elementBytes>UINT32_MAX)return false;
                    p.stride=plane.metadata.strideBytes?plane.metadata.strideBytes:uint32_t(elementBytes);
                }
                if(!s.Dispatch(1,b,p,p.outputWords,r))return false;
                s.outputs.push_back(std::move(output));return true;
            };
            for(auto const& plane:s.base->planes())if(!gather(plane,false))goto failed;
            for(auto const& plane:s.base->sourceFrames())if(!gather(plane,true))goto failed;
            if(!s.Submit(1,before,r))goto failed;
        }
        s.scatterSubmitted=true;
        if(result)*result=VK_SUCCESS;
        return true;
    failed:
        if(result)*result=r==VK_SUCCESS?VK_ERROR_INITIALIZATION_FAILED:r;
        return false;
    }catch(std::bad_alloc const&){if(result)*result=VK_ERROR_OUT_OF_HOST_MEMORY;return false;}
}
VkResult LengthCompactionPipeline::Candidate::PollScatter(uint32_t* semanticStatus) {
    auto& s=*state_;if(!s.scatterSubmitted)return VK_ERROR_INITIALIZATION_FAILED;
    if(!s.scatterProved){auto r=s.Prove(1);if(r!=VK_SUCCESS)return r;s.scatterProved=true;}
    if(semanticStatus)*semanticStatus=s.semantic;
    return VK_SUCCESS;
}
} // namespace usdGen::vulkan
