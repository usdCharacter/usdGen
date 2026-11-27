// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

// testUsdGenVulkanWidthOverlapWitness — device proof for the Vulkan port of
// gpu/widthOverlapWitness.cu (test-only stream overlap probe).
//
// Case V (host-only): the shared admission helper rejects short counters,
// expected<2, and zero dwell, mirroring LaunchWidthOverlapWitness.
// Case S (serial): two back-to-back single-queue probes observe arrivals 2
// with maxActive 1 and a balanced live count, proving serial detection.
// Case O (overlap, when the device exposes two compute queues): two probes
// submitted back-to-back on distinct queues both complete; arrivals is 2,
// the live count balances, and maxActive of 2 proves overlap was observed.
// A device that serializes still passes with maxActive 1 (the probe
// correctly reports what happened); the outcome is printed, never silent.
#include "usdGenCurves/vulkan/widthOverlapWitness.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"witness check failed: %s (%d)\n",#x,__LINE__); return 1; } } while(false)

using usdGen::vulkan::VulkanWidthOverlapWitnessControls;
using usdGen::vulkan::ValidateVulkanWidthOverlapWitnessArgs;
using usdGen::vulkan::kVulkanWidthOverlapWitnessCounters;

struct Buffer { VkBuffer b=VK_NULL_HANDLE; VkDeviceMemory m=VK_NULL_HANDLE; };

static uint32_t MemoryType(VkPhysicalDevice p, uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties m{}; vkGetPhysicalDeviceMemoryProperties(p,&m);
    for(uint32_t i=0;i<m.memoryTypeCount;++i) if((bits&(1u<<i)) && (m.memoryTypes[i].propertyFlags&want)==want) return i;
    return UINT32_MAX;
}
static bool MakeBuffer(VkDevice d, VkPhysicalDevice p, VkDeviceSize n, VkBufferUsageFlags usage,
                       VkMemoryPropertyFlags props, Buffer* out) {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO}; bi.size=std::max<VkDeviceSize>(n,4); bi.usage=usage; bi.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    if(vkCreateBuffer(d,&bi,nullptr,&out->b)!=VK_SUCCESS) return false; VkMemoryRequirements r{}; vkGetBufferMemoryRequirements(d,out->b,&r);
    uint32_t t=MemoryType(p,r.memoryTypeBits,props); if(t==UINT32_MAX) return false; VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO}; ai.allocationSize=r.size; ai.memoryTypeIndex=t;
    if(vkAllocateMemory(d,&ai,nullptr,&out->m)!=VK_SUCCESS) return false; return vkBindBufferMemory(d,out->b,out->m,0)==VK_SUCCESS;
}
static bool Upload(VkDevice d, Buffer const& b, void const* src, size_t n) { void* p=nullptr; if(vkMapMemory(d,b.m,0,n,0,&p)!=VK_SUCCESS)return false; std::memcpy(p,src,n); vkUnmapMemory(d,b.m); return true; }
static bool Read(VkDevice d, Buffer const& b, void* dst, size_t n) { void* p=nullptr; if(vkMapMemory(d,b.m,0,n,0,&p)!=VK_SUCCESS)return false; std::memcpy(dst,p,n); vkUnmapMemory(d,b.m); return true; }

int main(int argc,char**argv) {
    // Case V: host admission parity with LaunchWidthOverlapWitness. No GPU needed.
    CHECK(!ValidateVulkanWidthOverlapWitnessArgs(2, 2, 100));
    CHECK(!ValidateVulkanWidthOverlapWitnessArgs(3, 1, 100));
    CHECK(!ValidateVulkanWidthOverlapWitnessArgs(3, 0, 100));
    CHECK(!ValidateVulkanWidthOverlapWitnessArgs(3, 2, 0));
    CHECK(ValidateVulkanWidthOverlapWitnessArgs(3, 2, 1));
    CHECK(ValidateVulkanWidthOverlapWitnessArgs(4, 3, 100));
    std::printf("witness validation: 6/6 admission checks agree with CUDA\n");

    if(argc!=2) { std::fprintf(stderr,"usage: %s widthOverlapWitness.spv\n",argv[0]); return 1; }
    std::ifstream in(argv[1],std::ios::binary); CHECK(in); std::vector<char> raw((std::istreambuf_iterator<char>(in)),{}); CHECK(!raw.empty()&&raw.size()%4==0); std::vector<uint32_t> code(raw.size()/4); std::memcpy(code.data(),raw.data(),raw.size());
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.pApplicationName="usdGen Vulkan Witness"; app.apiVersion=VK_API_VERSION_1_2;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ii.pApplicationInfo=&app; VkInstance instance=VK_NULL_HANDLE; CHECK(vkCreateInstance(&ii,nullptr,&instance)==VK_SUCCESS);
    uint32_t np=0; CHECK(vkEnumeratePhysicalDevices(instance,&np,nullptr)==VK_SUCCESS&&np); std::vector<VkPhysicalDevice> ps(np); CHECK(vkEnumeratePhysicalDevices(instance,&np,ps.data())==VK_SUCCESS);
    VkPhysicalDevice physical=VK_NULL_HANDLE; uint32_t family=UINT32_MAX, familyQueues=0;
    // Prefer a compute-only family: this probe is pure compute, and a
    // graphics-capable family forces graphics-context allocation that can
    // fail under memory pressure even when compute channels are available.
    for(int pass=0;pass<2 && !physical;++pass) {
        for(auto p:ps) { VkPhysicalDeviceProperties pr{}; vkGetPhysicalDeviceProperties(p,&pr); if(pr.vendorID!=0x10de) continue; uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(p,&nq,nullptr); std::vector<VkQueueFamilyProperties> q(nq); vkGetPhysicalDeviceQueueFamilyProperties(p,&nq,q.data()); for(uint32_t i=0;i<nq;++i) { bool const compute=q[i].queueFlags&VK_QUEUE_COMPUTE_BIT; bool const graphics=q[i].queueFlags&VK_QUEUE_GRAPHICS_BIT; if(compute && (pass==1 || !graphics)){physical=p;family=i;familyQueues=q[i].queueCount;break;} } if(physical)break; }
    }
    if (!physical || family == UINT32_MAX) {
        std::fprintf(stderr,"NVIDIA Vulkan compute device unavailable\n");
        vkDestroyInstance(instance,nullptr);
        return 77;
    }
    VkPhysicalDeviceProperties selected{};
    vkGetPhysicalDeviceProperties(physical,&selected);
    std::printf("witness device: %s (family %u, %u queues)\n",selected.deviceName,family,familyQueues);
    uint32_t wantQueues = std::min(familyQueues, 2u);
    float priorities[2]={1,1}; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex=family; qi.queueCount=wantQueues; qi.pQueuePriorities=priorities;
    VkPhysicalDeviceFeatures features{}; VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.queueCreateInfoCount=1; di.pQueueCreateInfos=&qi; di.pEnabledFeatures=&features; VkDevice device=VK_NULL_HANDLE; CHECK(vkCreateDevice(physical,&di,nullptr,&device)==VK_SUCCESS);
    VkQueue queues[2]={VK_NULL_HANDLE,VK_NULL_HANDLE}; vkGetDeviceQueue(device,family,0,&queues[0]); if(wantQueues>1) vkGetDeviceQueue(device,family,1,&queues[1]);
    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex=family; VkCommandPool pool; CHECK(vkCreateCommandPool(device,&pci,nullptr,&pool)==VK_SUCCESS);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=code.size()*sizeof(uint32_t); smi.pCode=code.data(); VkShaderModule shader; CHECK(vkCreateShaderModule(device,&smi,nullptr,&shader)==VK_SUCCESS);
    VkDescriptorSetLayoutBinding bind{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr};
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; sli.bindingCount=1;sli.pBindings=&bind;VkDescriptorSetLayout sl;CHECK(vkCreateDescriptorSetLayout(device,&sli,nullptr,&sl)==VK_SUCCESS);
    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT,0,8}; VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&sl;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pc;VkPipelineLayout layout;CHECK(vkCreatePipelineLayout(device,&pli,nullptr,&layout)==VK_SUCCESS);
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};ss.stage=VK_SHADER_STAGE_COMPUTE_BIT;ss.module=shader;ss.pName="main";VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cpi.stage=ss;cpi.layout=layout;VkPipeline pipeline;CHECK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cpi,nullptr,&pipeline)==VK_SUCCESS);
    VkDescriptorPoolSize psize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1};VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=1;dpi.poolSizeCount=1;dpi.pPoolSizes=&psize;VkDescriptorPool dp;CHECK(vkCreateDescriptorPool(device,&dpi,nullptr,&dp)==VK_SUCCESS);VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};dai.descriptorPool=dp;dai.descriptorSetCount=1;dai.pSetLayouts=&sl;VkDescriptorSet ds;CHECK(vkAllocateDescriptorSets(device,&dai,&ds)==VK_SUCCESS);

    Buffer counters;
    auto const memory=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    size_t const bytes=kVulkanWidthOverlapWitnessCounters*sizeof(uint32_t);
    CHECK(MakeBuffer(device,physical,bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,memory,&counters));
    VkDescriptorBufferInfo info{counters.b,0,bytes};
    VkWriteDescriptorSet ws{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    ws.dstSet=ds; ws.dstBinding=0; ws.descriptorCount=1;
    ws.descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ws.pBufferInfo=&info;
    vkUpdateDescriptorSets(device,1,&ws,0,nullptr);

    auto record=[&](VulkanWidthOverlapWitnessControls controls, VkCommandBuffer* out)->bool {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool=pool; cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount=1;
        if(vkAllocateCommandBuffers(device,&cai,out)!=VK_SUCCESS) return false;
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if(vkBeginCommandBuffer(*out,&cbi)!=VK_SUCCESS) return false;
        VkMemoryBarrier uploadBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        uploadBarrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;
        uploadBarrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(*out,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,1,&uploadBarrier,0,nullptr,0,nullptr);
        vkCmdBindPipeline(*out,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
        vkCmdBindDescriptorSets(*out,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&ds,0,nullptr);
        vkCmdPushConstants(*out,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(controls),&controls);
        vkCmdDispatch(*out,1,1,1);
        VkMemoryBarrier readBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        readBarrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        readBarrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(*out,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,
                             0,1,&readBarrier,0,nullptr,0,nullptr);
        return vkEndCommandBuffer(*out)==VK_SUCCESS;
    };
    auto submit=[&](VkQueue queue, VkCommandBuffer cb, VkFence* fence)->bool {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount=1; si.pCommandBuffers=&cb;
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if(vkCreateFence(device,&fi,nullptr,fence)!=VK_SUCCESS) return false;
        return vkQueueSubmit(queue,1,&si,*fence)==VK_SUCCESS;
    };
    auto wait=[&](VkFence fence)->bool {
        return vkWaitForFences(device,1,&fence,VK_TRUE,60000000000ull)==VK_SUCCESS;
    };

    // Case S: two serial probes on one queue.
    {
        uint32_t zero[3]={0,0,0};
        CHECK(Upload(device,counters,zero,bytes));
        VulkanWidthOverlapWitnessControls controls{2, 1u<<20};
        CHECK(ValidateVulkanWidthOverlapWitnessArgs(3, controls.expected, controls.dwellIterations));
        for (int round=0; round<2; ++round) {
            VkCommandBuffer cb=VK_NULL_HANDLE; VkFence fence=VK_NULL_HANDLE;
            CHECK(record(controls,&cb) && submit(queues[0],cb,&fence) && wait(fence));
        }
        uint32_t out[3]={9,9,9};
        CHECK(Read(device,counters,out,bytes));
        std::printf("witness serial: live=%u maxActive=%u arrivals=%u\n",out[0],out[1],out[2]);
        CHECK(out[0]==0 && out[1]==1 && out[2]==2);
    }

    // Case O: two probes submitted back-to-back on distinct queues.
    if (wantQueues > 1) {
        uint32_t zero[3]={0,0,0};
        CHECK(Upload(device,counters,zero,bytes));
        VulkanWidthOverlapWitnessControls controls{2, 1u<<26};
        CHECK(ValidateVulkanWidthOverlapWitnessArgs(3, controls.expected, controls.dwellIterations));
        VkCommandBuffer cb0=VK_NULL_HANDLE, cb1=VK_NULL_HANDLE;
        VkFence f0=VK_NULL_HANDLE, f1=VK_NULL_HANDLE;
        CHECK(record(controls,&cb0) && record(controls,&cb1));
        CHECK(submit(queues[0],cb0,&f0) && submit(queues[1],cb1,&f1));
        CHECK(wait(f0) && wait(f1));
        uint32_t out[3]={9,9,9};
        CHECK(Read(device,counters,out,bytes));
        std::printf("witness overlap: live=%u maxActive=%u arrivals=%u\n",out[0],out[1],out[2]);
        CHECK(out[0]==0 && out[2]==2 && (out[1]==1 || out[1]==2));
        std::printf("witness overlap %s\n", out[1]==2 ? "OBSERVED" : "not observed (device serialized)");
    } else {
        std::printf("witness overlap: skipped (single queue)\n");
    }
    std::printf("witness PASS\n");
    return 0;
}
