// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include "nonWidthComparePipeline.h"
#include "nonWidthMetadata.h"
#include "sourceGeneration.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <new>

namespace usdGen::vulkan {
namespace {
constexpr VkDeviceSize kWordBytes = sizeof(uint32_t);
constexpr uint32_t kLocalSize = 256;
struct Copy { VkBuffer left, right; VkDeviceSize bytes, padded; };
bool Round(VkDeviceSize in, VkDeviceSize* out) {
    if (in > std::numeric_limits<VkDeviceSize>::max() - 3) return false;
    *out = (in + 3) & ~VkDeviceSize(3); return true;
}
bool Add(VkDeviceSize* total, VkDeviceSize n) {
    if (n > std::numeric_limits<VkDeviceSize>::max() - *total) return false;
    *total += n; return true;
}
bool IsWidth(VulkanSourceGeneration::PlaneView const& p) {
    return p.metadata.semantic == UsdGenDeviceChannelSemantic::Widths;
}
bool AddCopy(VulkanSourceGeneration::PlaneView const& l, VulkanSourceGeneration::PlaneView const& r,
    std::vector<Copy>* copies, VkDeviceSize* total) {
    if (IsWidth(l)) return true;
    if (l.bytes != r.bytes) return false;
    if (!l.bytes) return !l.buffer && !r.buffer;
    if (!l.buffer || !r.buffer) return false;
    VkDeviceSize padded = 0;
    if (!Round(l.bytes, &padded) || !Add(total, padded)) return false;
    copies->push_back({l.buffer, r.buffer, l.bytes, padded}); return true;
}
void Barrier(VkCommandBuffer c, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
    VkPipelineStageFlags dstStage, VkAccessFlags dstAccess) {
    VkMemoryBarrier b{}; b.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    b.srcAccessMask = srcAccess; b.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(c, srcStage, dstStage, 0, 1, &b, 0, nullptr, 0, nullptr);
}
} // namespace

struct NonWidthComparePipeline::Native {
    std::shared_ptr<DeviceContext> context;
    VkShaderModule shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptors = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    ~Native() { if (!context) return; auto d = context->device(); if (pipeline) vkDestroyPipeline(d,pipeline,nullptr); if (layout) vkDestroyPipelineLayout(d,layout,nullptr); if (descriptors) vkDestroyDescriptorSetLayout(d,descriptors,nullptr); if (shader) vkDestroyShaderModule(d,shader,nullptr); }
};
struct NonWidthComparePipeline::Candidate::State {
    std::shared_ptr<Native> native;
    std::shared_ptr<const VulkanSourceGeneration> left, right;
    std::shared_ptr<ChargedBuffer> leftPacked, rightPacked, status;
    VkDescriptorPool descriptors = VK_NULL_HANDLE;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    uint32_t semantic = 0;
    bool pending = false, lost = false, proved = false;
    std::unique_ptr<std::shared_ptr<State>> quarantine;
    ~State() { auto d=native->context->device(); if(fence)vkDestroyFence(d,fence,nullptr); if(commands)vkDestroyCommandPool(d,commands,nullptr); if(descriptors)vkDestroyDescriptorPool(d,descriptors,nullptr); }
};
NonWidthComparePipeline::NonWidthComparePipeline(std::shared_ptr<Native> n):native_(std::move(n)) {}
NonWidthComparePipeline::~NonWidthComparePipeline() = default;
std::shared_ptr<DeviceContext> const& NonWidthComparePipeline::context() const noexcept { return native_->context; }

std::shared_ptr<NonWidthComparePipeline> NonWidthComparePipeline::Create(std::shared_ptr<DeviceContext> context, std::vector<uint32_t> const& code, VkResult* result) {
    auto done=[&](VkResult r){if(result)*result=r;}; done(VK_ERROR_INITIALIZATION_FAILED);
    if(!context||code.size()<5||code.front()!=0x07230203u||code.size()>std::numeric_limits<size_t>::max()/4)return{};
    VkPhysicalDeviceProperties props{}; vkGetPhysicalDeviceProperties(context->physicalDevice(),&props);
    if(props.limits.maxComputeWorkGroupInvocations<kLocalSize||props.limits.maxComputeWorkGroupSize[0]<kLocalSize)return{};
    try { auto n=std::make_shared<Native>(); n->context=std::move(context); auto d=n->context->device(); VkResult r;
        VkShaderModuleCreateInfo sm{}; sm.sType=VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO; sm.codeSize=code.size()*4; sm.pCode=code.data(); r=vkCreateShaderModule(d,&sm,nullptr,&n->shader); if(r){done(r);return{};}
        VkDescriptorSetLayoutBinding b[3]{}; for(uint32_t i=0;i<3;++i){b[i].binding=i;b[i].descriptorCount=1;b[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;b[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT;}
        VkDescriptorSetLayoutCreateInfo di{}; di.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;di.bindingCount=3;di.pBindings=b;r=vkCreateDescriptorSetLayout(d,&di,nullptr,&n->descriptors);if(r){done(r);return{};}
        VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,4}; VkPipelineLayoutCreateInfo li{};li.sType=VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;li.setLayoutCount=1;li.pSetLayouts=&n->descriptors;li.pushConstantRangeCount=1;li.pPushConstantRanges=&push;r=vkCreatePipelineLayout(d,&li,nullptr,&n->layout);if(r){done(r);return{};}
        VkComputePipelineCreateInfo pi{};pi.sType=VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;pi.layout=n->layout;pi.stage.sType=VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;pi.stage.stage=VK_SHADER_STAGE_COMPUTE_BIT;pi.stage.module=n->shader;pi.stage.pName="main";r=vkCreateComputePipelines(d,VK_NULL_HANDLE,1,&pi,nullptr,&n->pipeline);if(r){done(r);return{};}done(VK_SUCCESS);return std::shared_ptr<NonWidthComparePipeline>(new NonWidthComparePipeline(std::move(n)));
    } catch(std::bad_alloc const&){done(VK_ERROR_OUT_OF_HOST_MEMORY);return{};}
}
NonWidthComparePipeline::Candidate::Candidate(std::shared_ptr<State> s):state_(std::move(s)) {}
NonWidthComparePipeline::Candidate::~Candidate(){if(state_&&state_->pending)Quarantine();}
void NonWidthComparePipeline::Candidate::Quarantine() noexcept { if(!state_||!state_->pending||state_->lost)return;state_->lost=true;state_->leftPacked->Quarantine();state_->rightPacked->Quarantine();state_->status->Quarantine();*state_->quarantine=state_;(void)state_->quarantine.release(); }
bool NonWidthComparePipeline::Candidate::succeeded()const noexcept{return state_&&!state_->lost&&state_->proved&&state_->semantic==0;}
std::shared_ptr<const VulkanSourceGeneration> NonWidthComparePipeline::Candidate::leftOwner()const noexcept{return state_?state_->left:nullptr;}
std::shared_ptr<const VulkanSourceGeneration> NonWidthComparePipeline::Candidate::rightOwner()const noexcept{return state_?state_->right:nullptr;}
VkResult NonWidthComparePipeline::Candidate::Poll(Status* out) {
    if (!state_) return VK_ERROR_INITIALIZATION_FAILED;
    auto& s = *state_;
    if (s.lost) { if (out) *out = Status::LostProof; return VK_ERROR_DEVICE_LOST; }
    if (!s.proved) {
        VkResult r = vkGetFenceStatus(s.native->context->device(), s.fence);
        if (r == VK_NOT_READY) { if (out) *out = Status::NotReady; return r; }
        if (r != VK_SUCCESS) { Quarantine(); if (out) *out = Status::LostProof; return r; }
        r = s.leftPacked->PollComplete();
        if (r == VK_SUCCESS) r = s.rightPacked->PollComplete();
        if (r == VK_SUCCESS) r = s.status->PollComplete();
        if (r != VK_SUCCESS) { Quarantine(); if (out) *out = Status::LostProof; return r; }
        void* p = nullptr;
        r = vkMapMemory(s.native->context->device(), s.status->memory(), 0, 4, 0, &p);
        if (r != VK_SUCCESS || !p) {
            Quarantine();
            if (out) *out = Status::LostProof;
            return r == VK_SUCCESS ? VK_ERROR_MEMORY_MAP_FAILED : r;
        }
        std::memcpy(&s.semantic, p, 4);
        vkUnmapMemory(s.native->context->device(), s.status->memory());
        s.pending = false;
        s.proved = true;
    }
    if (out) *out = Status::Ready;
    return VK_SUCCESS;
}

std::unique_ptr<NonWidthComparePipeline::Candidate> NonWidthComparePipeline::Begin(std::shared_ptr<const VulkanSourceGeneration> left,std::shared_ptr<const VulkanSourceGeneration> right,VkResult* result,BeforeSubmit beforeSubmit){
 auto done=[&](VkResult r){if(result)*result=r;};done(VK_ERROR_INITIALIZATION_FAILED);auto context=native_->context;
 if(!left||!right||left->context()!=context||right->context()!=context||!SameNonWidthMetadata(*left,*right))return{};
 try{std::vector<Copy> copies;copies.reserve(left->planes().size()+left->sourceFrames().size());VkDeviceSize total=0;
  // Width is optional on a source and may be appended by a COW child. Walk
  // the filtered sequences independently, exactly as metadata admission does.
  auto const& leftPlanes = left->planes();
  auto const& rightPlanes = right->planes();
  size_t li = 0, ri = 0;
  for (;;) {
      while (li != leftPlanes.size() && IsWidth(leftPlanes[li])) ++li;
      while (ri != rightPlanes.size() && IsWidth(rightPlanes[ri])) ++ri;
      if (li == leftPlanes.size() || ri == rightPlanes.size()) {
          if (li != leftPlanes.size() || ri != rightPlanes.size()) return {};
          break;
      }
      if (!AddCopy(leftPlanes[li++], rightPlanes[ri++], &copies, &total)) return {};
  }
  for(size_t i=0;i<left->sourceFrames().size();++i)if(!AddCopy(left->sourceFrames()[i],right->sourceFrames()[i],&copies,&total))return{};
  VkDeviceSize const descriptorBytes=std::max(kWordBytes,total);uint64_t const words64=total/kWordBytes,groups64=std::max(uint64_t(1),(words64+kLocalSize-1)/kLocalSize);VkPhysicalDeviceProperties props{};vkGetPhysicalDeviceProperties(context->physicalDevice(),&props);
  if(descriptorBytes>props.limits.maxStorageBufferRange||words64>UINT32_MAX||groups64>props.limits.maxComputeWorkGroupCount[0])return{};
  auto s=std::make_shared<Candidate::State>();s->native=native_;s->left=std::move(left);s->right=std::move(right);s->quarantine=std::make_unique<std::shared_ptr<Candidate::State>>();auto candidate=std::unique_ptr<Candidate>(new Candidate(s));auto d=context->device();VkResult r;
  VkBufferCreateInfo bi{};bi.sType=VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;bi.size=descriptorBytes;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT|VK_BUFFER_USAGE_TRANSFER_DST_BIT;bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;s->leftPacked=ChargedBuffer::Create(context,bi,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,UsdGenExecutionResourceKind::Scratch,&r);if(!s->leftPacked){done(r);return{};}s->rightPacked=ChargedBuffer::Create(context,bi,VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,UsdGenExecutionResourceKind::Scratch,&r);if(!s->rightPacked){done(r);return{};}
  bi.size=4;bi.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;s->status=ChargedBuffer::Create(context,bi,VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,UsdGenExecutionResourceKind::Scratch,&r);if(!s->status){done(r);return{};}void* mapped=nullptr;r=vkMapMemory(d,s->status->memory(),0,4,0,&mapped);if(r!=VK_SUCCESS||!mapped){done(r==VK_SUCCESS?VK_ERROR_MEMORY_MAP_FAILED:r);return{};}uint32_t zero=0;std::memcpy(mapped,&zero,4);vkUnmapMemory(d,s->status->memory());
  VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3};VkDescriptorPoolCreateInfo pool{};pool.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;pool.maxSets=1;pool.poolSizeCount=1;pool.pPoolSizes=&poolSize;r=vkCreateDescriptorPool(d,&pool,nullptr,&s->descriptors);if(r){done(r);return{};}VkDescriptorSetAllocateInfo allocation{};allocation.sType=VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;allocation.descriptorPool=s->descriptors;allocation.descriptorSetCount=1;allocation.pSetLayouts=&native_->descriptors;VkDescriptorSet set=VK_NULL_HANDLE;r=vkAllocateDescriptorSets(d,&allocation,&set);if(r){done(r);return{};}
  VkDescriptorBufferInfo infos[3]={{s->leftPacked->buffer(),0,descriptorBytes},{s->rightPacked->buffer(),0,descriptorBytes},{s->status->buffer(),0,4}};VkWriteDescriptorSet writes[3]{};for(uint32_t i=0;i<3;++i){writes[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;writes[i].dstSet=set;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&infos[i];}vkUpdateDescriptorSets(d,3,writes,0,nullptr);
  VkCommandPoolCreateInfo poolInfo{};poolInfo.sType=VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;poolInfo.queueFamilyIndex=context->computeQueueFamily();r=vkCreateCommandPool(d,&poolInfo,nullptr,&s->commands);if(r){done(r);return{};}VkCommandBufferAllocateInfo ai{};ai.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;ai.commandPool=s->commands;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=1;VkCommandBuffer command=VK_NULL_HANDLE;r=vkAllocateCommandBuffers(d,&ai,&command);if(r){done(r);return{};}VkCommandBufferBeginInfo begin{};begin.sType=VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;r=vkBeginCommandBuffer(command,&begin);if(r){done(r);return{};}
  Barrier(command,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_SHADER_WRITE_BIT|VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_READ_BIT);vkCmdFillBuffer(command,s->leftPacked->buffer(),0,descriptorBytes,0);vkCmdFillBuffer(command,s->rightPacked->buffer(),0,descriptorBytes,0);Barrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);VkDeviceSize offset=0;for(Copy const& copy:copies){VkBufferCopy region{0,offset,copy.bytes};vkCmdCopyBuffer(command,copy.left,s->leftPacked->buffer(),1,&region);vkCmdCopyBuffer(command,copy.right,s->rightPacked->buffer(),1,&region);offset+=copy.padded;}
  Barrier(command,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_HOST_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT);Barrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_READ_BIT);vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,native_->pipeline);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,native_->layout,0,1,&set,0,nullptr);uint32_t words=uint32_t(words64);vkCmdPushConstants(command,native_->layout,VK_SHADER_STAGE_COMPUTE_BIT,0,4,&words);vkCmdDispatch(command,uint32_t(groups64),1,1);Barrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_ACCESS_SHADER_WRITE_BIT,VK_PIPELINE_STAGE_HOST_BIT,VK_ACCESS_HOST_READ_BIT);r=vkEndCommandBuffer(command);if(r){done(r);return{};}
  VkFenceCreateInfo fi{};fi.sType=VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;r=vkCreateFence(d,&fi,nullptr,&s->fence);if(r){done(r);return{};}if(beforeSubmit){bool admitted=false;try{admitted=beforeSubmit();}catch(...){done(VK_ERROR_UNKNOWN);return{};}if(!admitted){done(VK_ERROR_OUT_OF_DEVICE_MEMORY);return{};}}r=s->leftPacked->MarkSubmitted(s->fence,s);if(r==VK_SUCCESS)r=s->rightPacked->MarkSubmitted(s->fence,s);if(r==VK_SUCCESS)r=s->status->MarkSubmitted(s->fence,s);if(r!=VK_SUCCESS){s->pending=true;candidate->Quarantine();done(r);return{};}s->pending=true;VkSubmitInfo submit{};submit.sType=VK_STRUCTURE_TYPE_SUBMIT_INFO;submit.commandBufferCount=1;submit.pCommandBuffers=&command;r=vkQueueSubmit(context->computeQueue(),1,&submit,s->fence);if(r!=VK_SUCCESS){candidate->Quarantine();done(r);return{};}done(VK_SUCCESS);return candidate;
 }catch(std::bad_alloc const&){done(VK_ERROR_OUT_OF_HOST_MEMORY);return{};}
}
} // namespace usdGen::vulkan
