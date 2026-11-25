// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "widthBlendPipeline.h"
#include <cmath>
#include <cstring>
#include <new>
namespace usdGen::vulkan {
    struct WidthBlendPipeline::Native {
        std::shared_ptr<DeviceContext> context;
        VkShaderModule shader=VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptors=VK_NULL_HANDLE;
        VkPipelineLayout layout=VK_NULL_HANDLE;
        VkPipeline pipeline=VK_NULL_HANDLE;
        ~Native(){
            if(!context)return;
            auto d=context->device();
            if(pipeline)vkDestroyPipeline(d,pipeline,nullptr);
            if(layout)vkDestroyPipelineLayout(d,layout,nullptr);
            if(descriptors)vkDestroyDescriptorSetLayout(d,descriptors,nullptr);
            if(shader)vkDestroyShaderModule(d,shader,nullptr);
        }
    }
    ;
    struct WidthBlendPipeline::Candidate::State {
        std::shared_ptr<Native> native;
        std::shared_ptr<const ChargedBuffer> left,right;
        std::shared_ptr<ChargedBuffer> output,status;
        VkDescriptorPool descriptors=VK_NULL_HANDLE;
        VkCommandPool commands=VK_NULL_HANDLE;
        VkFence fence=VK_NULL_HANDLE;
        uint32_t count=0,semantic=UINT32_MAX;
        bool pending=false,lost=false,proved=false;
        std::unique_ptr<std::shared_ptr<State>> quarantine;
        ~State(){
            auto d=native->context->device();
            if(fence)vkDestroyFence(d,fence,nullptr);
            if(commands)vkDestroyCommandPool(d,commands,nullptr);
            if(descriptors)vkDestroyDescriptorPool(d,descriptors,nullptr);
        }
    }
    ;
    WidthBlendPipeline::WidthBlendPipeline(std::shared_ptr<Native> n):native_(std::move(n)){
    }
    WidthBlendPipeline::~WidthBlendPipeline()=default;
    std::shared_ptr<DeviceContext> const& WidthBlendPipeline::context()const noexcept{
        return native_->context;
    }
    std::shared_ptr<WidthBlendPipeline> WidthBlendPipeline::Create(std::shared_ptr<DeviceContext> context,std::vector<uint32_t> const& code,VkResult* out){
        auto finish=[&](VkResult r){
            if(out)*out=r;
        }
        ;
        finish(VK_ERROR_INITIALIZATION_FAILED);
        if(!context||code.size()<5||code.front()!=0x07230203u)return{};
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(context->physicalDevice(),&p);
        if(p.limits.maxComputeWorkGroupInvocations<256||p.limits.maxComputeWorkGroupSize[0]<256)return{};
        try{
            auto n=std::make_shared<Native>();
            n->context=std::move(context);
            auto d=n->context->device();
            VkShaderModuleCreateInfo sm{};
            sm.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            sm.codeSize=code.size()*sizeof(uint32_t);
            sm.pCode=code.data();
            VkResult r=vkCreateShaderModule(d,&sm,nullptr,&n->shader);
            if(r){
                finish(r);
                return{};
            }
            VkDescriptorSetLayoutBinding b[4]{};
            for(uint32_t i=0;i<4;i++){
                b[i].binding=i;
                b[i].descriptorCount=1;
                b[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                b[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;
            }
            VkDescriptorSetLayoutCreateInfo ds{};
            ds.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            ds.bindingCount=4;
            ds.pBindings=b;
            r=vkCreateDescriptorSetLayout(d,&ds,nullptr,&n->descriptors);
            if(r){
                finish(r);
                return{};
            }
            VkPushConstantRange pc{
                VK_SHADER_STAGE_COMPUTE_BIT,0,8
            }
            ;
            VkPipelineLayoutCreateInfo pl{};
            pl.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl.setLayoutCount=1;
            pl.pSetLayouts=&n->descriptors;
            pl.pushConstantRangeCount=1;
            pl.pPushConstantRanges=&pc;
            r=vkCreatePipelineLayout(d,&pl,nullptr,&n->layout);
            if(r){
                finish(r);
                return{};
            }
            VkComputePipelineCreateInfo cp{};
            cp.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
            cp.layout=n->layout;
            cp.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            cp.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;
            cp.stage.module=n->shader;
            cp.stage.pName="main";
            r=vkCreateComputePipelines(d,VK_NULL_HANDLE,1,&cp,nullptr,&n->pipeline);
            if(r){
                finish(r);
                return{};
            }
            finish(VK_SUCCESS);
            return std::shared_ptr<WidthBlendPipeline>(new WidthBlendPipeline(std::move(n)));
        }
        catch(std::bad_alloc const&){
            finish(VK_ERROR_OUT_OF_HOST_MEMORY);
            return{};
        }
    }
    WidthBlendPipeline::Candidate::Candidate(std::shared_ptr<State>s):state_(std::move(s)){
    }
    WidthBlendPipeline::Candidate::~Candidate(){
        if(state_&&state_->pending)Quarantine();
    }
    void WidthBlendPipeline::Candidate::Quarantine()noexcept{
        if(!state_||!state_->pending||state_->lost)return;
        state_->lost=true;
        *state_->quarantine=state_;
        (void)state_->quarantine.release();
    }
    VkResult WidthBlendPipeline::Candidate::Poll(uint32_t* status){
        if(!state_)return VK_ERROR_INITIALIZATION_FAILED;
        auto&s=*state_;
        if(s.lost)return VK_ERROR_DEVICE_LOST;
        if(!s.proved){
            VkResult r=vkGetFenceStatus(s.native->context->device(),s.fence);
            if(r==VK_NOT_READY)return r;
            if(r!=VK_SUCCESS){
                Quarantine();
                return r;
            }
            s.pending=false;
            void*p=nullptr;
            r=vkMapMemory(s.native->context->device(),s.status->memory(),0,4,0,&p);
            if(r!=VK_SUCCESS)return r;
            std::memcpy(&s.semantic,p,4);
            vkUnmapMemory(s.native->context->device(),s.status->memory());
            s.proved=true;
        }
        if(status)*status=s.semantic;
        return VK_SUCCESS;
    }
    std::shared_ptr<const ChargedBuffer> WidthBlendPipeline::Candidate::output()const noexcept{
        return state_&&state_->proved&&!state_->semantic?state_->output:nullptr;
    }
    std::shared_ptr<const ChargedBuffer> WidthBlendPipeline::Candidate::leftOwner()const noexcept{
        return state_?state_->left:nullptr;
    }
    std::shared_ptr<const ChargedBuffer> WidthBlendPipeline::Candidate::rightOwner()const noexcept{
        return state_?state_->right:nullptr;
    }
    uint32_t WidthBlendPipeline::Candidate::count()const noexcept{
        return state_?state_->count:0;
    }
    std::shared_ptr<DeviceContext> WidthBlendPipeline::Candidate::context()const noexcept{
        return state_?state_->native->context:nullptr;
    }
    bool WidthBlendPipeline::Candidate::succeeded()const noexcept{
        return state_&&!state_->lost&&state_->proved&&!state_->semantic;
    }
    std::unique_ptr<WidthBlendPipeline::Candidate> WidthBlendPipeline::Begin(std::shared_ptr<const ChargedBuffer> left,std::shared_ptr<const ChargedBuffer> right,uint32_t count,float blend,VkResult*out,BeforeSubmit hook){
        auto finish=[&](VkResult r){
            if(out)*out=r;
        }
        ;
        finish(VK_ERROR_INITIALIZATION_FAILED);
        auto context=native_->context;
        VkDeviceSize bytes=VkDeviceSize(count)*sizeof(float);
        if(!std::isfinite(blend)||blend<0||blend>1||(count&&(!left||!right))||(left&&(left->context()!=context||left->unproven()||!left->buffer()||!(left->usage()&VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)||left->sizeBytes()!=bytes))||(right&&(right->context()!=context||right->unproven()||!right->buffer()||!(right->usage()&VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)||right->sizeBytes()!=bytes)))return{};
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(context->physicalDevice(),&p);
        uint64_t groups=(uint64_t(count)+255)/256;
        if(groups>p.limits.maxComputeWorkGroupCount[0]||bytes>p.limits.maxStorageBufferRange)return{};
        try{
            auto s=std::make_shared<Candidate::State>();
            s->native=native_;
            s->left=std::move(left);
            s->right=std::move(right);
            s->count=count;
            s->quarantine=std::make_unique<std::shared_ptr<Candidate::State>>();
            auto c=std::unique_ptr<Candidate>(new Candidate(s));
            if(!count){
                s->proved=true;
                s->semantic=0;
                finish(VK_SUCCESS);
                return c;
            }
            auto d=context->device();
            VkBufferCreateInfo bi{};
            bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            bi.size=bytes;
            bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
            VkResult r;
            s->output=ChargedBuffer::Create(context,bi,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,UsdGenExecutionResourceKind::Active,&r);
            if(!s->output){
                finish(r);
                return{};
            }
            bi.size=4;
            bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
            s->status=ChargedBuffer::Create(context,bi,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,UsdGenExecutionResourceKind::Scratch,&r);
            if(!s->status){
                finish(r);
                return{};
            }
            void*data=nullptr;
            r=vkMapMemory(d,s->status->memory(),0,4,0,&data);
            if(r){
                finish(r);
                return{};
            }
            std::memset(data,0,4);
            vkUnmapMemory(d,s->status->memory());
            VkDescriptorPoolSize ps{
                VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4
            }
            ;
            VkDescriptorPoolCreateInfo dp{};
            dp.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            dp.maxSets=1;
            dp.poolSizeCount=1;
            dp.pPoolSizes=&ps;
            r=vkCreateDescriptorPool(d,&dp,nullptr,&s->descriptors);
            if(r){
                finish(r);
                return{};
            }
            VkDescriptorSetAllocateInfo ai{};
            ai.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            ai.descriptorPool=s->descriptors;
            ai.descriptorSetCount=1;
            ai.pSetLayouts=&native_->descriptors;
            VkDescriptorSet set;
            r=vkAllocateDescriptorSets(d,&ai,&set);
            if(r){
                finish(r);
                return{};
            }
            VkDescriptorBufferInfo info[4]={
                {
                    s->left->buffer(),0,bytes
                }
                ,{
                    s->right->buffer(),0,bytes
                }
                ,{
                    s->output->buffer(),0,bytes
                }
                ,{
                    s->status->buffer(),0,4
                }
            }
            ;
            VkWriteDescriptorSet w[4]{};
            for(uint32_t i=0;i<4;i++){
                w[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                w[i].dstSet=set;
                w[i].dstBinding=i;
                w[i].descriptorCount=1;
                w[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                w[i].pBufferInfo=&info[i];
            }
            vkUpdateDescriptorSets(d,4,w,0,nullptr);
            VkCommandPoolCreateInfo pi{};
            pi.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pi.queueFamilyIndex=context->computeQueueFamily();
            r=vkCreateCommandPool(d,&pi,nullptr,&s->commands);
            if(r){
                finish(r);
                return{};
            }
            VkCommandBufferAllocateInfo ci{};
            ci.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            ci.commandPool=s->commands;
            ci.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            ci.commandBufferCount=1;
            VkCommandBuffer cmd;
            r=vkAllocateCommandBuffers(d,&ci,&cmd);
            if(r){
                finish(r);
                return{};
            }
            VkCommandBufferBeginInfo begin{};
            begin.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            r=vkBeginCommandBuffer(cmd,&begin);
            if(r){
                finish(r);
                return{};
            }
            VkMemoryBarrier before{};
            before.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            before.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT|VK_ACCESS_SHADER_WRITE_BIT;
            before.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&before,0,nullptr,0,nullptr);
            vkCmdBindPipeline(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,native_->pipeline);
            vkCmdBindDescriptorSets(cmd,VK_PIPELINE_BIND_POINT_COMPUTE,native_->layout,0,1,&set,0,nullptr);
            struct C{
                uint32_t count;
                float blend;
            }
            ;
            static_assert(sizeof(C)==8,"width blend shader ABI");
            C controls{
                count,blend
            }
            ;
            vkCmdPushConstants(cmd,native_->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,8,&controls);
            vkCmdDispatch(cmd,uint32_t(groups),1,1);
            VkMemoryBarrier after{};
            after.sType=VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            after.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
            after.dstAccessMask=VK_ACCESS_HOST_READ_BIT|VK_ACCESS_TRANSFER_READ_BIT|VK_ACCESS_SHADER_READ_BIT;
            vkCmdPipelineBarrier(cmd,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT|VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,0,1,&after,0,nullptr,0,nullptr);
            r=vkEndCommandBuffer(cmd);
            if(r){
                finish(r);
                return{};
            }
            VkFenceCreateInfo fi{};
            fi.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            r=vkCreateFence(d,&fi,nullptr,&s->fence);
            if(r){
                finish(r);
                return{};
            }
            if(hook){
                bool admitted=false;
                try{
                    admitted=hook();
                }
                catch(...){
                    finish(VK_ERROR_UNKNOWN);
                    return{};
                }
                if(!admitted){
                    finish(VK_ERROR_OUT_OF_DEVICE_MEMORY);
                    return{};
                }
            }
            VkSubmitInfo sub{};
            sub.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;
            sub.commandBufferCount=1;
            sub.pCommandBuffers=&cmd;
            s->pending=true;
            r=vkQueueSubmit(context->computeQueue(),1,&sub,s->fence);
            if(r){
                c->Quarantine();
                finish(r);
                return{};
            }
            finish(VK_SUCCESS);
            return c;
        }
        catch(std::bad_alloc const&){
            finish(VK_ERROR_OUT_OF_HOST_MEMORY);
            return{};
        }
    }
}
// namespace usdGen::vulkan
