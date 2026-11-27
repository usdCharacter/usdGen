// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#include <vulkan/vulkan.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"Vulkan width check failed: %s (%d)\n",#x,__LINE__); return 1; } } while(false)
struct Controls { uint32_t count; float width; uint32_t replace; };
static_assert(sizeof(Controls) == 12, "Width push-constant ABI");
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
static void FreeBuffer(VkDevice d, Buffer& b) { if(b.b) vkDestroyBuffer(d,b.b,nullptr); if(b.m) vkFreeMemory(d,b.m,nullptr); b={}; }
static bool Upload(VkDevice d, Buffer const& b, void const* src, size_t n) { void* p=nullptr; if(vkMapMemory(d,b.m,0,n,0,&p)!=VK_SUCCESS)return false; std::memcpy(p,src,n); vkUnmapMemory(d,b.m); return true; }
static bool Read(VkDevice d, Buffer const& b, void* dst, size_t n) { void* p=nullptr; if(vkMapMemory(d,b.m,0,n,0,&p)!=VK_SUCCESS)return false; std::memcpy(dst,p,n); vkUnmapMemory(d,b.m); return true; }
int main(int argc,char**argv) {
    if(argc!=2) { std::fprintf(stderr,"usage: %s width.spv\n",argv[0]); return 1; }
    std::ifstream in(argv[1],std::ios::binary); CHECK(in); std::vector<char> raw((std::istreambuf_iterator<char>(in)),{}); CHECK(!raw.empty()&&raw.size()%4==0); std::vector<uint32_t> code(raw.size()/4); std::memcpy(code.data(),raw.data(),raw.size());
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO}; app.pApplicationName="usdGen Vulkan Width"; app.apiVersion=VK_API_VERSION_1_2;
    VkInstanceCreateInfo ii{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ii.pApplicationInfo=&app; VkInstance instance=VK_NULL_HANDLE; CHECK(vkCreateInstance(&ii,nullptr,&instance)==VK_SUCCESS);
    uint32_t np=0; CHECK(vkEnumeratePhysicalDevices(instance,&np,nullptr)==VK_SUCCESS&&np); std::vector<VkPhysicalDevice> ps(np); CHECK(vkEnumeratePhysicalDevices(instance,&np,ps.data())==VK_SUCCESS);
    VkPhysicalDevice physical=VK_NULL_HANDLE; uint32_t family=UINT32_MAX;
    for(auto p:ps) { VkPhysicalDeviceProperties pr{}; vkGetPhysicalDeviceProperties(p,&pr); if(pr.vendorID!=0x10de) continue; uint32_t nq=0; vkGetPhysicalDeviceQueueFamilyProperties(p,&nq,nullptr); std::vector<VkQueueFamilyProperties> q(nq); vkGetPhysicalDeviceQueueFamilyProperties(p,&nq,q.data()); for(uint32_t i=0;i<nq;++i) if(q[i].queueFlags&VK_QUEUE_COMPUTE_BIT){physical=p;family=i;break;} if(physical)break; }
    if (!physical || family == UINT32_MAX) {
        std::fprintf(stderr,"NVIDIA Vulkan compute device unavailable\n");
        vkDestroyInstance(instance,nullptr);
        return 77;
    }
    VkPhysicalDeviceProperties selected{};
    vkGetPhysicalDeviceProperties(physical,&selected);
    std::printf("Vulkan Width device: %s\n",selected.deviceName);
    float priority=1; VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}; qi.queueFamilyIndex=family; qi.queueCount=1; qi.pQueuePriorities=&priority;
    VkPhysicalDeviceFeatures features{}; VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO}; di.queueCreateInfoCount=1; di.pQueueCreateInfos=&qi; di.pEnabledFeatures=&features; VkDevice device=VK_NULL_HANDLE; CHECK(vkCreateDevice(physical,&di,nullptr,&device)==VK_SUCCESS);
    VkQueue queue=VK_NULL_HANDLE; vkGetDeviceQueue(device,family,0,&queue); VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO}; pci.queueFamilyIndex=family; VkCommandPool pool; CHECK(vkCreateCommandPool(device,&pci,nullptr,&pool)==VK_SUCCESS);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO}; smi.codeSize=code.size()*sizeof(uint32_t); smi.pCode=code.data(); VkShaderModule shader; CHECK(vkCreateShaderModule(device,&smi,nullptr,&shader)==VK_SUCCESS);
    VkDescriptorSetLayoutBinding binds[3]={{0,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{1,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr},{2,VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,1,VK_SHADER_STAGE_COMPUTE_BIT,nullptr}};
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO}; sli.bindingCount=3;sli.pBindings=binds;VkDescriptorSetLayout sl;CHECK(vkCreateDescriptorSetLayout(device,&sli,nullptr,&sl)==VK_SUCCESS);
    VkPushConstantRange pc{VK_SHADER_STAGE_COMPUTE_BIT,0,12}; VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};pli.setLayoutCount=1;pli.pSetLayouts=&sl;pli.pushConstantRangeCount=1;pli.pPushConstantRanges=&pc;VkPipelineLayout layout;CHECK(vkCreatePipelineLayout(device,&pli,nullptr,&layout)==VK_SUCCESS);
    VkPipelineShaderStageCreateInfo ss{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};ss.stage=VK_SHADER_STAGE_COMPUTE_BIT;ss.module=shader;ss.pName="main";VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};cpi.stage=ss;cpi.layout=layout;VkPipeline pipeline;CHECK(vkCreateComputePipelines(device,VK_NULL_HANDLE,1,&cpi,nullptr,&pipeline)==VK_SUCCESS);
    VkDescriptorPoolSize psizes[1]={{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,3}};VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};dpi.maxSets=1;dpi.poolSizeCount=1;dpi.pPoolSizes=psizes;VkDescriptorPool dp;CHECK(vkCreateDescriptorPool(device,&dpi,nullptr,&dp)==VK_SUCCESS);VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};dai.descriptorPool=dp;dai.descriptorSetCount=1;dai.pSetLayouts=&sl;VkDescriptorSet ds;CHECK(vkAllocateDescriptorSets(device,&dai,&ds)==VK_SUCCESS);
    auto run=[&](std::vector<float> values,float width,uint32_t replace,uint32_t expectedError)->bool {
        size_t const n=values.size();
        if(!n) return std::isfinite(width) && width>=0 && replace<=1;
        if(n>UINT32_MAX || (n+255)/256>selected.limits.maxComputeWorkGroupCount[0]) return false;
        Buffer ib,ob,sb;
        auto const memory=VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        size_t const bytes=(n+1)*sizeof(float);
        if(!MakeBuffer(device,physical,bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,memory,&ib)||
           !MakeBuffer(device,physical,bytes,VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,memory,&ob)||
           !MakeBuffer(device,physical,sizeof(uint32_t),VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,memory,&sb)) return false;
        uint32_t zero=0;
        std::vector<float> input=values, sentinel(n+1,12345.f);
        input.push_back(-9876.f);
        if(!Upload(device,ib,input.data(),bytes)||!Upload(device,ob,sentinel.data(),bytes)||
           !Upload(device,sb,&zero,sizeof(zero))) return false;
        VkDescriptorBufferInfo infos[3]={{ib.b,0,bytes},{ob.b,0,bytes},{sb.b,0,sizeof(zero)}};
        VkWriteDescriptorSet ws[3]{};
        for(uint32_t i=0;i<3;++i) {
            ws[i].sType=VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            ws[i].dstSet=ds; ws[i].dstBinding=i; ws[i].descriptorCount=1;
            ws[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; ws[i].pBufferInfo=&infos[i];
        }
        vkUpdateDescriptorSets(device,3,ws,0,nullptr);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool=pool; cai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount=1;
        VkCommandBuffer cb=VK_NULL_HANDLE;
        if(vkAllocateCommandBuffers(device,&cai,&cb)!=VK_SUCCESS) return false;
        VkCommandBufferBeginInfo cbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        if(vkBeginCommandBuffer(cb,&cbi)!=VK_SUCCESS) return false;
        VkMemoryBarrier uploadBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        uploadBarrier.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;
        uploadBarrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0,1,&uploadBarrier,0,nullptr,0,nullptr);
        vkCmdBindPipeline(cb,VK_PIPELINE_BIND_POINT_COMPUTE,pipeline);
        vkCmdBindDescriptorSets(cb,VK_PIPELINE_BIND_POINT_COMPUTE,layout,0,1,&ds,0,nullptr);
        Controls controls{uint32_t(n),width,replace};
        vkCmdPushConstants(cb,layout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(controls),&controls);
        vkCmdDispatch(cb,(uint32_t(n)+255)/256,1,1);
        VkMemoryBarrier readBarrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        readBarrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;
        readBarrier.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cb,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_HOST_BIT,
                             0,1,&readBarrier,0,nullptr,0,nullptr);
        if(vkEndCommandBuffer(cb)!=VK_SUCCESS) return false;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount=1; si.pCommandBuffers=&cb;
        VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VkFence fence=VK_NULL_HANDLE;
        if(vkCreateFence(device,&fi,nullptr,&fence)!=VK_SUCCESS) return false;
        if(vkQueueSubmit(queue,1,&si,fence)!=VK_SUCCESS||
           vkWaitForFences(device,1,&fence,VK_TRUE,10000000000ull)!=VK_SUCCESS) {
            // No completion proof: return to main's failing CHECK without
            // destroying any possibly in-flight handles. Process exit owns
            // this isolated-test quarantine; never reuse it in production.
            std::fprintf(stderr,"Vulkan submission completion unproven\n");
            return false;
        }
        uint32_t error=99;
        std::vector<float> out(n+1), retained(n+1);
        bool ok=Read(device,sb,&error,sizeof(error))&&Read(device,ib,retained.data(),bytes)&&
            Read(device,ob,out.data(),bytes);
        ok=ok&&error==expectedError&&std::memcmp(input.data(),retained.data(),bytes)==0&&
            out.back()==sentinel.back();
        if(!expectedError) for(size_t i=0;i<n;++i) {
            float expected=replace?values[i]+(width-values[i]):values[i]*(1.f+(width-1.f));
            ok=ok&&out[i]==expected;
        }
        // Failed output is initialized but deliberately never accepted as a
        // generation or interpreted as valid geometry.
        vkDestroyFence(device,fence,nullptr);
        vkFreeCommandBuffers(device,pool,1,&cb);
        FreeBuffer(device,ib); FreeBuffer(device,ob); FreeBuffer(device,sb);
        return ok;
    };
    CHECK(run({1,2,3},4,1,0)); CHECK(run(std::vector<float>(257,2),3,0,0)); CHECK(run({1},-1,1,1)); CHECK(run({std::nanf("")},2,1,1));
    // Zero points are admitted by the host contract and must skip dispatch.
    CHECK(run({},4,1,0));
    CHECK(!run({},-1,1,0)); CHECK(!run({},1,2,0));
    CHECK(run({1,2},1,2,1));
    CHECK(run({3.4e38f},3.4e38f,0,2));
    std::printf("Vulkan Width dispatch/COW/status checks: PASS\n");
    vkDestroyDescriptorPool(device,dp,nullptr);vkDestroyPipeline(device,pipeline,nullptr);vkDestroyPipelineLayout(device,layout,nullptr);vkDestroyDescriptorSetLayout(device,sl,nullptr);vkDestroyShaderModule(device,shader,nullptr);vkDestroyCommandPool(device,pool,nullptr);vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);return 0;
}
