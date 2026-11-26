// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// testUsdGenVulkanExprVkParity — Vulkan expression (SeExpr) parity.
//
// Proves the Vulkan expression executor (vulkan/exprVk*.cpp + exprVk*.comp)
// evaluates every SeExpr program with CUDA-equivalent results, citing the
// CUDA test's own contract (testUsdGenCudaExpressionParity.cpp):
//   * the exact IEEE-754 set (arithmetic, compare/select/min/max/clamp,
//     noise/hash/cellnoise/voronoi/curve/spline/choose/pick, vectors,
//     colours): BIT-IDENTICAL with the CPU reference lane, hence with CUDA;
//   * the libm transcendentals (sin cos tan asin acos atan sinh cosh tanh
//     exp log log10 pow cbrt atan2, and bias/contrast/gaussstep/remap/angle/
//     rotate/up/midhsi which reach one): the Vulkan executor evaluates those
//     steps on the host with the same <cmath>/exprMath.h calls the CPU lane
//     makes on bit-identical inputs, so Vulkan is BIT-IDENTICAL with the CPU
//     lane there too, hence within the CUDA test's 1e-12 of CUDA.
// When a CUDA device is available the test also compares Vulkan directly
// against the CUDA twin (gpu::CudaExpressionContext + CudaExpressionProgram)
// under the same agreement rule. CUDA initialization, upload or evaluation
// failure on an available device fails the gate. Without an available CUDA
// device the Vulkan-only evidence is reported and the gate returns 77.
//
// Also proves: context field parity (every materialized variable, point and
// primitive domains, with and without root frames), every Store destination
// type bitwise, expression-driven Width application vs the same-order CPU
// formula and vs the literal width.comp lane on constants, UsdGenExprOp
// application (displacement/width x perCV/perCurve, literal/connected mask),
// and the InvalidValue/InvalidGeometry/InvalidChannel fail-closed paths.
//
// Usage:
//   testUsdGenVulkanExprVkParity <evaluate.spv> <context.spv> <width.spv>
//       <exprop.spv> [literalWidth.spv] [--device nvidia|llvmpipe]
// Device selection prefers NVIDIA (vkCreateDevice -3/OUT_OF_MEMORY retries
// 5x with 60s waits: tenant load), then falls back to llvmpipe as secondary
// evidence and reports NVIDIA as retry-pending.

#include "usdGenExpr/vulkan/exprVkApply.h"
#include "usdGenUsd/lower.h"
#include "usdGenExpr/vulkan/exprVkContext.h"
#include "usdGenExpr/vulkan/exprVkEvaluate.h"
#include "usdGenExpr/vulkan/exprVkParameters.h"
#include "usdGenExpr/vulkan/exprVkPlan.h"
#include "usdGenExpr/vulkan/exprVkProgram.h"
#include "usdGen/vulkan/widthPipeline.h"
#include "usdGen/expressions/cpuEvaluator.h"
#include "usdGen/expressions/frontend.h"

#include <vulkan/vulkan.h>

#ifdef USDGEN_ENABLE_CUDA
#include "gpu/deviceBuffer.h"
#include "gpu/expression.h"
#include "gpu/expressionContext.h"
#include <cuda_runtime.h>
#include <vector_types.h>
#endif

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

using namespace usdGen;

// Unfused host oracle (testUsdGenVulkanExprVkParityOracle.cpp, compiled
// -ffp-contract=off -fno-fast-math): same admission and per-element loop as
// expr::EvaluateProgram, but the shared interpreter templates instantiate
// without FMA contraction, which is what the `precise` shader evaluates.
namespace usdGen::expr {
CpuExpressionStatus EvaluateProgramUnfused(IRProgram const &program,
                                           CpuExpressionInputs const &inputs,
                                           CpuExpressionOutput const &output);
}

namespace {

int g_failures = 0;
int g_cudaSkipped = 0;
void Check(bool ok, std::string const& what) {
    if (ok) {
        std::printf("PASS: %s\n", what.c_str());
    } else {
        ++g_failures;
        std::printf("FAIL: %s\n", what.c_str());
    }
}

// Four strands of six CVs, leaning like the CUDA test's scene so arc length,
// hairT and root UVs are all non-trivial. Copied from
// testUsdGenCudaExpressionParity.cpp so the two scenes agree element for
// element.
struct Scene {
    static constexpr unsigned kCurves = 4, kCvs = 6, kPoints = kCurves * kCvs;
    std::vector<float> px, py, pz, hairT, widths;
    std::vector<float> rest, rootUV;
    std::vector<uint64_t> ids;
    std::vector<uint32_t> offsets;

    Scene() {
        for (unsigned c = 0; c < kCurves; ++c) {
            const float u = float(c) / float(kCurves - 1);
            const float v = 0.25f * float(c);
            rootUV.push_back(u);
            rootUV.push_back(v);
            ids.push_back((uint64_t(c + 1) << 33) + 11u * c + 3u);
            offsets.push_back(c * kCvs);
            for (unsigned i = 0; i < kCvs; ++i) {
                const float t = float(i) / float(kCvs - 1);
                const float x = u + 0.35f * 0.3f * t;
                const float y = 0.3f * t * (1.0f + 0.1f * float(c));
                const float z = v + 0.2f * 0.3f * t;
                px.push_back(x);
                py.push_back(y);
                pz.push_back(z);
                rest.push_back(x);
                rest.push_back(y);
                rest.push_back(z);
                hairT.push_back(t);
                widths.push_back(0.012f);
            }
        }
        offsets.push_back(kPoints);
    }

    expr::CpuCurveGeometryView Host() const {
        expr::CpuCurveGeometryView view;
        view.px = px.data();
        view.py = py.data();
        view.pz = pz.data();
        view.rest = rest.data();
        view.widths = widths.data();
        view.hairT = hairT.data();
        view.rootUV = rootUV.data();
        view.stableIds = ids.data();
        view.curveOffsets = offsets.data();
        view.curveCount = kCurves;
        view.pointCount = kPoints;
        return view;
    }
};

expr::Context TestControls(expr::Domain domain) {
    expr::Context controls;
    controls.frame = 1.0;
    controls.time = 1.0 / 24.0;
    controls.seed = 41;
    controls.descId = expr::DescriptionId("/World/Groom/Fur");
    controls.domain = domain;
    return controls;
}

// How closely two lanes must agree (the CUDA test's own contract).
enum class Agreement { Exact, Transcendental };

bool CloseEnough(double a, double b) {
    uint64_t abits = 0, bbits = 0;
    std::memcpy(&abits, &a, 8);
    std::memcpy(&bbits, &b, 8);
    if (abits == bbits) return true;
    if (!(a == a) || !(b == b)) return false;
    const double scale = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
    return std::fabs(a - b) <= 1e-12 * (scale > 1.0 ? scale : 1.0);
}

// --- device setup ----------------------------------------------------------

struct VkNative {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    int physicalIndex = -1;
    std::string name;
    bool isNvidia = false;
    ~VkNative() {
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

// Creates a Vulkan 1.2 compute device with shaderFloat64+shaderInt64. NVIDIA
// is preferred; vkCreateDevice -3 (OUT_OF_DEVICE_MEMORY, tenant load)
// retries up to 5x with 60s waits, then llvmpipe serves as secondary
// evidence with *nvidiaPending set.
std::shared_ptr<VkNative> CreateDevice(bool forceLlvmpipe, bool* nvidiaPending,
                                       bool* unavailable) {
    *nvidiaPending = false;
    *unavailable = false;
    auto native = std::make_shared<VkNative>();
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "usdGen exprVk parity";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo create{};
    create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create.pApplicationInfo = &app;
    if (vkCreateInstance(&create, nullptr, &native->instance) != VK_SUCCESS) return {};
    uint32_t count = 0;
    if (vkEnumeratePhysicalDevices(native->instance, &count, nullptr) != VK_SUCCESS) return {};
    std::vector<VkPhysicalDevice> devices(count);
    if (vkEnumeratePhysicalDevices(native->instance, &count, devices.data()) != VK_SUCCESS)
        return {};
    int nvidia = -1, lvp = -1;
    uint32_t nvidiaFamily = UINT32_MAX, lvpFamily = UINT32_MAX;
    for (uint32_t index = 0; index < count; ++index) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devices[index], &props);
        if (props.apiVersion < VK_API_VERSION_1_2) continue;
        VkPhysicalDeviceFeatures features{};
        vkGetPhysicalDeviceFeatures(devices[index], &features);
        if (features.shaderFloat64 != VK_TRUE || features.shaderInt64 != VK_TRUE) continue;
        uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, nullptr);
        std::vector<VkQueueFamilyProperties> info(families);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[index], &families, info.data());
        uint32_t family = UINT32_MAX;
        for (uint32_t f = 0; f < families; ++f) {
            if (!info[f].queueCount || !(info[f].queueFlags & VK_QUEUE_COMPUTE_BIT)) continue;
            family = f;
            break;
        }
        if (family == UINT32_MAX) continue;
        std::string name = props.deviceName;
        bool const isNvidia = props.vendorID == 0x10de;
        bool const isLvp = name.find("lvp") != std::string::npos ||
            name.find("llvmpipe") != std::string::npos ||
            name.find("Lavapipe") != std::string::npos;
        if (isNvidia && nvidia < 0) {
            nvidia = int(index);
            nvidiaFamily = family;
        }
        if (isLvp && lvp < 0) {
            lvp = int(index);
            lvpFamily = family;
        }
    }
    auto makeDevice = [&](int index, uint32_t family, bool isNvidia) {
        float priority = 1.f;
        VkDeviceQueueCreateInfo queue{};
        queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue.queueFamilyIndex = family;
        queue.queueCount = 1;
        queue.pQueuePriorities = &priority;
        VkPhysicalDeviceFeatures core{};
        core.shaderFloat64 = VK_TRUE;
        core.shaderInt64 = VK_TRUE;
        VkDeviceCreateInfo device{};
        device.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device.pEnabledFeatures = &core;
        device.queueCreateInfoCount = 1;
        device.pQueueCreateInfos = &queue;
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devices[index], &props);
        for (int attempt = 0;; ++attempt) {
            VkResult r =
                vkCreateDevice(devices[index], &device, nullptr, &native->device);
            if (r == VK_SUCCESS) {
                native->physical = devices[index];
                native->physicalIndex = index;
                native->family = family;
                native->name = props.deviceName;
                native->isNvidia = isNvidia;
                vkGetDeviceQueue(native->device, family, 0, &native->queue);
                return true;
            }
            // -3 (INITIALIZATION_FAILED) and -2 (OUT_OF_DEVICE_MEMORY) are
            // tenant load, never a code defect: wait and retry.
            if (isNvidia &&
                (r == VK_ERROR_INITIALIZATION_FAILED ||
                 r == VK_ERROR_OUT_OF_DEVICE_MEMORY) &&
                attempt < 5) {
                std::printf("vkCreateDevice %d (attempt %d/5), waiting 60s...\n",
                            r, attempt + 1);
                std::this_thread::sleep_for(std::chrono::seconds(60));
                continue;
            }
            return false;
        }
    };
    if (!forceLlvmpipe && nvidia >= 0) {
        if (makeDevice(nvidia, nvidiaFamily, true)) return native;
        *nvidiaPending = true;
        std::printf("NVIDIA device creation failed after retries; "
                    "NVIDIA verdict retry-pending.\n");
    }
    char const* hardware = std::getenv("USDGEN_VULKAN_REQUIRE_HARDWARE");
    bool const requireHardware = hardware && std::strcmp(hardware, "1") == 0;
    if (lvp >= 0 && !requireHardware) {
        native->device = VK_NULL_HANDLE;
        if (makeDevice(lvp, lvpFamily, false)) {
            std::printf("SECONDARY EVIDENCE device: %s\n", native->name.c_str());
            return native;
        }
    }
    if (nvidia < 0 && (lvp < 0 || requireHardware)) *unavailable = true;
    return {};
}

std::vector<uint32_t> LoadSpv(char const* path, bool* ok) {
    *ok = false;
    std::ifstream shader(path, std::ios::binary);
    if (!shader) return {};
    std::vector<char> bytes((std::istreambuf_iterator<char>(shader)), {});
    if (bytes.empty() || bytes.size() % sizeof(uint32_t) != 0) return {};
    std::vector<uint32_t> code(bytes.size() / sizeof(uint32_t));
    std::memcpy(code.data(), bytes.data(), bytes.size());
    *ok = true;
    return code;
}

// --- buffer helpers ---------------------------------------------------------

std::shared_ptr<vulkan::ChargedBuffer> Upload(
    std::shared_ptr<vulkan::DeviceContext> const& context, void const* data,
    size_t bytes) {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = bytes ? bytes : 4;
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult r;
    auto buffer = vulkan::ChargedBuffer::Create(context, info,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        UsdGenExecutionResourceKind::Active, &r);
    if (!buffer) return {};
    void* mapped = nullptr;
    if (vkMapMemory(context->device(), buffer->memory(), 0, info.size, 0, &mapped) !=
        VK_SUCCESS)
        return {};
    if (bytes && data) std::memcpy(mapped, data, bytes);
    vkUnmapMemory(context->device(), buffer->memory());
    return buffer;
}

struct VkGeometry {
    vulkan::ExprVkCurveGeometry geometry;
    vulkan::ExprVkGeometryChannels channels;
};

VkGeometry UploadScene(std::shared_ptr<vulkan::DeviceContext> const& context,
                       Scene const& scene, bool* ok) {
    *ok = false;
    VkGeometry uploaded;
    std::vector<float> points;
    for (size_t i = 0; i < Scene::kPoints; ++i) {
        points.push_back(scene.px[i]);
        points.push_back(scene.py[i]);
        points.push_back(scene.pz[i]);
    }
    uploaded.geometry.points =
        Upload(context, points.data(), points.size() * sizeof(float));
    uploaded.geometry.restPoints =
        Upload(context, scene.rest.data(), scene.rest.size() * sizeof(float));
    uploaded.geometry.widths =
        Upload(context, scene.widths.data(), scene.widths.size() * sizeof(float));
    uploaded.geometry.curveOffsets =
        Upload(context, scene.offsets.data(), scene.offsets.size() * sizeof(uint32_t));
    uploaded.geometry.stableIds =
        Upload(context, scene.ids.data(), scene.ids.size() * sizeof(uint64_t));
    uploaded.geometry.curveCount = Scene::kCurves;
    uploaded.geometry.pointCount = Scene::kPoints;
    uploaded.channels.hairT =
        Upload(context, scene.hairT.data(), scene.hairT.size() * sizeof(float));
    uploaded.channels.rootUV =
        Upload(context, scene.rootUV.data(), scene.rootUV.size() * sizeof(float));
    *ok = uploaded.geometry.points && uploaded.geometry.restPoints &&
        uploaded.geometry.widths && uploaded.geometry.curveOffsets &&
        uploaded.geometry.stableIds && uploaded.channels.hairT && uploaded.channels.rootUV;
    return uploaded;
}

// Reads a host-visible buffer whole.
std::vector<uint8_t> Download(std::shared_ptr<vulkan::DeviceContext> const& context,
                              vulkan::ChargedBuffer const& buffer, bool* ok) {
    *ok = false;
    std::vector<uint8_t> bytes(buffer.sizeBytes());
    void* mapped = nullptr;
    if (vkMapMemory(context->device(), buffer.memory(), 0, buffer.sizeBytes(), 0, &mapped) !=
        VK_SUCCESS)
        return {};
    std::memcpy(bytes.data(), mapped, bytes.size());
    vkUnmapMemory(context->device(), buffer.memory());
    *ok = true;
    return bytes;
}

// Reads a device-local buffer through a staging copy on the test's queue.
std::vector<uint8_t> DownloadDeviceLocal(VkNative const& native,
                                         vulkan::ChargedBuffer const& buffer,
                                         bool* ok) {
    *ok = false;
    std::vector<uint8_t> bytes;
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = buffer.sizeBytes();
    info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer staging = VK_NULL_HANDLE;
    if (vkCreateBuffer(native.device, &info, nullptr, &staging) != VK_SUCCESS) return {};
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(native.device, staging, &requirements);
    VkPhysicalDeviceMemoryProperties memory{};
    vkGetPhysicalDeviceMemoryProperties(native.physical, &memory);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        bool const suitable = (requirements.memoryTypeBits & (1u << i)) != 0;
        auto const flags = memory.memoryTypes[i].propertyFlags;
        bool const host = (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0 &&
            (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
        if (suitable && host) {
            type = i;
            break;
        }
    }
    if (type == UINT32_MAX) {
        vkDestroyBuffer(native.device, staging, nullptr);
        return {};
    }
    VkDeviceMemory stagingMemory = VK_NULL_HANDLE;
    VkMemoryAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocate.allocationSize = requirements.size;
    allocate.memoryTypeIndex = type;
    if (vkAllocateMemory(native.device, &allocate, nullptr, &stagingMemory) != VK_SUCCESS) {
        vkDestroyBuffer(native.device, staging, nullptr);
        return {};
    }
    if (vkBindBufferMemory(native.device, staging, stagingMemory, 0) != VK_SUCCESS) {
        vkFreeMemory(native.device, stagingMemory, nullptr);
        vkDestroyBuffer(native.device, staging, nullptr);
        return {};
    }
    VkCommandPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.queueFamilyIndex = native.family;
    VkCommandPool commands = VK_NULL_HANDLE;
    VkCommandBuffer command = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    bool submitted = false;
    if (vkCreateCommandPool(native.device, &pool, nullptr, &commands) == VK_SUCCESS) {
        VkCommandBufferAllocateInfo allocateBuffer{};
        allocateBuffer.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocateBuffer.commandPool = commands;
        allocateBuffer.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocateBuffer.commandBufferCount = 1;
        if (vkAllocateCommandBuffers(native.device, &allocateBuffer, &command) ==
            VK_SUCCESS) {
            VkCommandBufferBeginInfo begin{};
            begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            if (vkBeginCommandBuffer(command, &begin) == VK_SUCCESS) {
                VkBufferCopy copy{0, 0, buffer.sizeBytes()};
                vkCmdCopyBuffer(command, buffer.buffer(), staging, 1, &copy);
                if (vkEndCommandBuffer(command) == VK_SUCCESS) {
                    VkFenceCreateInfo fenceInfo{};
                    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                    if (vkCreateFence(native.device, &fenceInfo, nullptr, &fence) ==
                        VK_SUCCESS) {
                        VkSubmitInfo submit{};
                        submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                        submit.commandBufferCount = 1;
                        submit.pCommandBuffers = &command;
                        submitted =
                            vkQueueSubmit(native.queue, 1, &submit, fence) == VK_SUCCESS;
                    }
                }
            }
        }
    }
    bool waited = submitted &&
        vkWaitForFences(native.device, 1, &fence, VK_TRUE, 30000000000ull) == VK_SUCCESS;
    if (waited) {
        void* mapped = nullptr;
        if (vkMapMemory(native.device, stagingMemory, 0, info.size, 0, &mapped) ==
            VK_SUCCESS) {
            bytes.assign(static_cast<uint8_t*>(mapped),
                         static_cast<uint8_t*>(mapped) + info.size);
            vkUnmapMemory(native.device, stagingMemory);
            *ok = true;
        }
    }
    if (fence) vkDestroyFence(native.device, fence, nullptr);
    if (commands) vkDestroyCommandPool(native.device, commands, nullptr);
    vkFreeMemory(native.device, stagingMemory, nullptr);
    vkDestroyBuffer(native.device, staging, nullptr);
    return bytes;
}

// --- CPU oracle -------------------------------------------------------------

// Runs source on the CPU reference lane into doubles (any destination width).
bool RunCpu(std::string const& source, expr::Domain domain, expr::ScalarType type,
            unsigned components, VtValue const& literalValue, Scene const& scene,
            std::vector<double>* values, std::string* diagnostic) {
    expr::CpuExpressionContext built;
    expr::CpuCurveGeometryView view = scene.Host();
    expr::Context controls = TestControls(domain);
    std::string contextError;
    if (built.Build(view, controls, &contextError) != expr::CpuExpressionStatus::Ok) {
        if (diagnostic) *diagnostic = "cpu context: " + contextError;
        return false;
    }
    expr::FrontendOptions options;
    options.domain = domain;
    options.destination = type;
    options.components = components;
    expr::CompileResult compiled = expr::Frontend::Compile(source, options);
    if (!compiled.ok) {
        if (diagnostic) *diagnostic = "compile failed";
        return false;
    }
    expr::IRProgram const ir = compiled.program.IR();
    std::vector<double> literal(components, 0.0);
    if (type == expr::ScalarType::Float64 && literalValue.IsHolding<double>())
        literal[0] = literalValue.UncheckedGet<double>();
    else if (type == expr::ScalarType::Float32 && literalValue.IsHolding<float>())
        literal[0] = literalValue.UncheckedGet<float>();
    else if (literalValue.IsHolding<double>() && components == 1)
        literal[0] = literalValue.UncheckedGet<double>();
    else if (literalValue.IsHolding<float>() && components == 1)
        literal[0] = literalValue.UncheckedGet<float>();
    expr::CpuExpressionInputs inputs = built.Inputs();
    size_t const n = domain == expr::Domain::Groom ? 1
        : domain == expr::Domain::Primitive        ? Scene::kCurves
                                                  : Scene::kPoints;
    expr::CpuExpressionField value;
    value.data = literal.data();
    value.count = 1;
    value.domain = expr::Domain::Groom;
    value.components = components;
    inputs.fields[static_cast<unsigned>(expr::Variable::Value)] = value;
    inputs.count = n;
    // Float64 output is the bit microscope regardless of the IR destination.
    std::vector<double> out(n * components, 0.0);
    expr::CpuExpressionOutput output;
    output.data = out.data();
    output.count = n;
    output.type = expr::ScalarType::Float64;
    output.components = components;
    if (expr::EvaluateProgramUnfused(ir, inputs, output) != expr::CpuExpressionStatus::Ok) {
        if (diagnostic) *diagnostic = "cpu evaluate failed";
        return false;
    }
    *values = std::move(out);
    return true;
}

#ifdef USDGEN_ENABLE_CUDA
// --- CUDA twin (mirrors testUsdGenCudaExpressionParity.cpp) ------------------

struct CudaDeviceScene {
    gpu::DeviceBuffer<float3> points, rest;
    gpu::DeviceBuffer<float> widths, hairT;
    gpu::DeviceBuffer<float2> rootUV;
    gpu::DeviceBuffer<uint32_t> offsets;
    gpu::DeviceBuffer<uint64_t> ids;

    bool Upload(Scene const& scene) {
        std::vector<float3> p(Scene::kPoints), r(Scene::kPoints);
        std::vector<float2> uv(Scene::kCurves);
        for (unsigned i = 0; i < Scene::kPoints; ++i) {
            p[i] = make_float3(scene.px[i], scene.py[i], scene.pz[i]);
            r[i] = make_float3(
                scene.rest[i * 3], scene.rest[i * 3 + 1], scene.rest[i * 3 + 2]);
        }
        for (unsigned c = 0; c < Scene::kCurves; ++c)
            uv[c] = make_float2(scene.rootUV[c * 2], scene.rootUV[c * 2 + 1]);
        return points.reset(Scene::kPoints) == cudaSuccess &&
            rest.reset(Scene::kPoints) == cudaSuccess &&
            widths.reset(Scene::kPoints) == cudaSuccess &&
            hairT.reset(Scene::kPoints) == cudaSuccess &&
            rootUV.reset(Scene::kCurves) == cudaSuccess &&
            offsets.reset(Scene::kCurves + 1) == cudaSuccess &&
            ids.reset(Scene::kCurves) == cudaSuccess &&
            cudaMemcpy(points.data(), p.data(), p.size() * sizeof(float3),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(rest.data(), r.data(), r.size() * sizeof(float3),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(widths.data(), scene.widths.data(),
                       scene.widths.size() * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(hairT.data(), scene.hairT.data(),
                       scene.hairT.size() * sizeof(float),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(rootUV.data(), uv.data(), uv.size() * sizeof(float2),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(offsets.data(), scene.offsets.data(),
                       scene.offsets.size() * sizeof(uint32_t),
                       cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(ids.data(), scene.ids.data(), scene.ids.size() * sizeof(uint64_t),
                       cudaMemcpyHostToDevice) == cudaSuccess;
    }
};

struct CudaTwin {
    cudaStream_t stream = nullptr;
    CudaDeviceScene device;
    bool ready = false;
    bool unavailable = false;
    std::string note = "CUDA twin unavailable";
    ~CudaTwin() {
        if (stream) cudaStreamDestroy(stream);
    }
};

std::shared_ptr<CudaTwin> SetupCudaTwin(Scene const& scene) {
    auto twin = std::make_shared<CudaTwin>();
    int count = 0;
    cudaError_t const status = cudaGetDeviceCount(&count);
    twin->unavailable = status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
        (status == cudaSuccess && count == 0);
    if (twin->unavailable) {
        twin->note = "no CUDA device";
        return twin;
    }
    if (status != cudaSuccess) {
        twin->note = std::string("CUDA enumeration failed: ") + cudaGetErrorString(status);
        return twin;
    }
    if (cudaSetDevice(0) != cudaSuccess || cudaStreamCreate(&twin->stream) != cudaSuccess) {
        twin->note = "CUDA context creation failed";
        return twin;
    }
    if (!twin->device.Upload(scene)) {
        twin->note = "CUDA scene upload failed";
        return twin;
    }
    twin->ready = true;
    twin->note.clear();
    return twin;
}

bool RunCuda(CudaTwin& twin, std::string const& source, expr::Domain domain,
             double literalValue, std::vector<double>* values) {
    size_t const n = domain == expr::Domain::Groom ? 1
        : domain == expr::Domain::Primitive        ? Scene::kCurves
                                                  : Scene::kPoints;
    values->assign(n, 0.0);
    gpu::DeviceCurveGeometryView geometry{{twin.device.points.data(), Scene::kPoints},
        {twin.device.rest.data(), Scene::kPoints},
        {twin.device.widths.data(), Scene::kPoints},
        {twin.device.offsets.data(), Scene::kCurves + 1},
        {twin.device.ids.data(), Scene::kCurves}, Scene::kCurves, Scene::kPoints};
    gpu::CudaExpressionContext built;
    if (built.Build(geometry,
                    {{twin.device.hairT.data(), Scene::kPoints},
                        {twin.device.rootUV.data(), Scene::kCurves}},
                    TestControls(domain), twin.stream) !=
            gpu::ExpressionContextStatus::Ok ||
        built.Finish(twin.stream) != gpu::ExpressionContextStatus::Ok)
        return false;
    expr::FrontendOptions options;
    options.domain = domain;
    options.destination = expr::ScalarType::Float32;
    options.components = 1;
    expr::CompileResult compiled = expr::Frontend::Compile(source, options);
    if (!compiled.ok) return false;
    gpu::DeviceBuffer<double> deviceLiteral, deviceOutput;
    if (deviceLiteral.reset(1) != cudaSuccess || deviceOutput.reset(n) != cudaSuccess ||
        cudaMemcpy(deviceLiteral.data(), &literalValue, sizeof(literalValue),
                   cudaMemcpyHostToDevice) != cudaSuccess)
        return false;
    gpu::CudaExpressionProgram program;
    if (program.Upload(compiled.program.IR(), twin.stream) != gpu::ExpressionStatus::Ok)
        return false;
    gpu::ExpressionInputs inputs = built.Inputs();
    inputs.fields[static_cast<unsigned>(expr::Variable::Value)] = {deviceLiteral.data(), 1,
        expr::Domain::Groom, 1};
    if (program.Evaluate(inputs, {deviceOutput.data(), n, expr::ScalarType::Float64, 1},
                         twin.stream) != gpu::ExpressionStatus::Ok ||
        program.Finish(twin.stream) != gpu::ExpressionStatus::Ok ||
        cudaMemcpy(values->data(), deviceOutput.data(), n * sizeof(double),
                   cudaMemcpyDeviceToHost) != cudaSuccess)
        return false;
    return true;
}
#endif

// --- comparison ---------------------------------------------------------------

struct Lane {
    std::shared_ptr<vulkan::DeviceContext> context;
    vulkan::ExprVkParameterEvaluator* evaluator = nullptr;
    VkGeometry const* uploaded = nullptr;
    Scene const* scene = nullptr;
#ifdef USDGEN_ENABLE_CUDA
    CudaTwin* twin = nullptr;
#endif
};

UsdGenGraphDesc BindingGraph(std::string const& source, std::string const& nativeType,
                             expr::ScalarType scalar, unsigned components,
                             expr::Domain /*domain*/, VtValue const& /*literal*/) {
    UsdGenGraphDesc graph;
    UsdGenExpressionDesc expression;
    expression.path = SdfPath("/expr");
    expression.source = source;
    UsdGenExpressionOutputDesc output;
    output.name = TfToken("result");
    output.nativeType = TfToken(nativeType);
    output.shape.scalar = scalar;
    output.shape.elementCount = 1;
    output.shape.components = components;
    output.shape.rows = 1;
    output.shape.columns = 1;
    output.shape.isArray = false;
    expression.outputs.push_back(output);
    graph.expressions.push_back(expression);
    return graph;
}

UsdGenExpressionBinding MakeBinding(expr::Domain domain, VtValue const& literal,
                                    std::string const& nativeType,
                                    expr::ScalarType scalar, unsigned components,
                                    std::string const& destination) {
    UsdGenExpressionBinding binding;
    binding.expression = SdfPath("/expr");
    binding.output = TfToken("result");
    binding.nativeType = TfToken(nativeType);
    binding.destination = TfToken(destination);
    binding.destinationShape.scalar = scalar;
    binding.destinationShape.elementCount = 1;
    binding.destinationShape.components = components;
    binding.destinationShape.rows = 1;
    binding.destinationShape.columns = 1;
    binding.destinationShape.isArray = false;
    binding.domain = domain;
    binding.literal = literal;
    return binding;
}

// The IR must not depend on the destination width: the Float64 microscope
// legs compile Float64 while operators consume Float32, and both lanes must
// run the same instructions.
bool SameInstructions(expr::IRProgram const& a, expr::IRProgram const& b) {
    if (a.instructions.size() != b.instructions.size() || a.result != b.result ||
        a.outputCount != b.outputCount || a.registerCount != b.registerCount)
        return false;
    for (unsigned i = 0; i < 4; ++i)
        if (a.output[i] != b.output[i]) return false;
    for (size_t i = 0; i < a.instructions.size(); ++i) {
        auto const& x = a.instructions[i];
        auto const& y = b.instructions[i];
        uint64_t xi = 0, yi = 0;
        std::memcpy(&xi, &x.immediate, 8);
        std::memcpy(&yi, &y.immediate, 8);
        if (x.op != y.op || x.dst != y.dst || x.a != y.a || x.b != y.b || x.c != y.c ||
            x.variable != y.variable || x.compare != y.compare || xi != yi ||
            x.component != y.component)
            return false;
    }
    return true;
}

void CompareFloat64(Lane& lane, std::string const& label, std::string const& source,
                    expr::Domain domain, Agreement agreement, double literal = 0.012) {
    UsdGenGraphDesc graph =
        BindingGraph(source, "double", expr::ScalarType::Float64, 1, domain, VtValue(0.0));
    UsdGenExpressionBinding binding = MakeBinding(
        domain, VtValue(literal), "double", expr::ScalarType::Float64, 1, "width");
    vulkan::ExprVkCompiledBinding compiled;
    std::vector<std::string> diagnostics;
    if (vulkan::ExprVkCompileBinding(usdGenUsd::Lower(graph), usdGenUsd::Lower(binding), true, true, &compiled, &diagnostics) !=
        vulkan::ExprVkCompileStatus::Ok) {
        Check(false, label + ": compiles - " +
                (diagnostics.empty() ? std::string() : diagnostics.front()));
        return;
    }
    // Destination-independence of the IR (methodology guard).
    {
        expr::FrontendOptions f32;
        f32.domain = domain;
        f32.destination = expr::ScalarType::Float32;
        f32.components = 1;
        expr::CompileResult c32 = expr::Frontend::Compile(source, f32);
        if (!c32.ok || !SameInstructions(c32.program.IR(), compiled.ir)) {
            Check(false, label + ": IR destination independence");
            return;
        }
    }
    std::vector<vulkan::ExprVkCompiledBinding> bindings{compiled};
    auto status = lane.evaluator->Evaluate(
        bindings, lane.uploaded->geometry, lane.uploaded->channels,
        TestControls(domain), &diagnostics);
    if (status != vulkan::ExprVkParameterStatus::Ok) {
        Check(false, label + ": evaluates - " +
                (diagnostics.empty() ? std::string() : diagnostics.front()));
        return;
    }
    auto const* field = lane.evaluator->Find(TfToken("width"));
    if (!field || !field->data) {
        Check(false, label + ": publishes field");
        return;
    }
    bool ok = false;
    std::vector<uint8_t> bytes = Download(lane.context, *field->data, &ok);
    if (!ok) {
        Check(false, label + ": readback");
        return;
    }
    std::vector<double> cpu;
    std::string cpuError;
    if (!RunCpu(source, domain, expr::ScalarType::Float64, 1, VtValue(literal),
                *lane.scene, &cpu, &cpuError)) {
        Check(false, label + ": cpu oracle - " + cpuError);
        return;
    }
    size_t const n = cpu.size();
    if (bytes.size() != n * 8) {
        Check(false, label + ": slot count");
        return;
    }
    // Vulkan-vs-CPU is bit-identical for every program (exact prefixes are
    // IEEE on both; host steps run identical code on identical inputs).
    bool bits = true;
    for (size_t i = 0; i < n; ++i) {
        uint64_t slot = 0, expect = 0;
        std::memcpy(&slot, bytes.data() + i * 8, 8);
        std::memcpy(&expect, &cpu[i], 8);
        if (slot != expect) {
            if (bits) {
                double v = 0;
                std::memcpy(&v, &slot, 8);
                std::printf("  %s: slot %zu vulkan %a cpu %a\n", label.c_str(), i,
                            v, cpu[i]);
            }
            bits = false;
            if (i > 3) break;
        }
    }
    Check(bits, label + ": vulkan bit-identical vs cpu");
#ifdef USDGEN_ENABLE_CUDA
    if (lane.twin && lane.twin->ready) {
        std::vector<double> cuda;
        if (!RunCuda(*lane.twin, source, domain, literal, &cuda) ||
            cuda.size() != n) {
            Check(false, label + ": CUDA twin evaluation/readback failed");
            return;
        }
        // The CUDA test's own contract: exact bit-identical, transcendental
        // within 1e-12.
        bool agree = true;
        for (size_t i = 0; i < n; ++i) {
            double v = 0;
            std::memcpy(&v, bytes.data() + i * 8, 8);
            bool const same = agreement == Agreement::Exact
                ? std::memcmp(&v, &cuda[i], 8) == 0
                : CloseEnough(v, cuda[i]);
            if (!same) {
                if (agree)
                    std::printf("  %s: slot %zu vulkan %a cuda %a\n", label.c_str(), i,
                                v, cuda[i]);
                agree = false;
                if (i > 3) break;
            }
        }
        Check(agree, label + ": vulkan vs cuda (" +
                std::string(agreement == Agreement::Exact ? "exact" : "1e-12") + ")");
    } else {
        ++g_cudaSkipped;
    }
#else
    (void)agreement;
#endif
}

// The CUDA test's expression matrix, verbatim, so the Vulkan legs prove the
// same coverage it does (plus the groom leg the CUDA test lacks).
void ExpressionMatrix(Lane& lane) {
    using D = expr::Domain;
    CompareFloat64(lane, "width-point",
                   "$value * fit(smoothstep($t, 0.0, 1.0), 0, 1, 1.0, 0.15)",
                   D::Point, Agreement::Exact, 0.012);
    CompareFloat64(
        lane, "mask-prim", "$value * $u", D::Primitive, Agreement::Exact, 1.0);
    CompareFloat64(lane, "arithmetic",
                   "clamp(($P[0] + $Pref[1] - $rootP[2]) / max($cLength, 0.001), -4, 4) + "
                   "min($primIndex, $pointIndex) * abs($t - 0.5) + sqrt($t) + "
                   "floor($idLo * 0.001) + fmod($pointCount, 4) + "
                   "mix($u, $v, $t) + ($t > 0.5 ? $cWidth : -$cWidth)",
                   D::Point, Agreement::Exact, 2.0);
    CompareFloat64(lane, "scalar",
                   "round($P[0] * 7.3) + trunc($P[1] * -5.1) + hypot($P[0], $P[2]) + "
                   "($t % 0.3) + deg($t) + rad($t) + invert($t) + ceil($t * 3) + "
                   "compress($t, 2, 5) + expand($t, 0.1, 0.9) + boxstep($t, 0.5) + "
                   "linearstep($t, 0.2, 0.8) + smoothstep($t, 0.8, 0.2) + "
                   "remap($t, 0.5, 0.2, 0.3, 0) + remap($t, 0.5, 0.2, 0.3, 1) + "
                   "cycle($pointIndex, 1, 3) + fit($t, 0, 1, -2, 3)",
                   D::Point, Agreement::Exact, 1.0);
    CompareFloat64(lane, "vector-colour",
                   "dist($P, $rootP) + length($P) + dot($P, $Pref) + cross($P, $Pref)[1] + "
                   "norm($P)[2] + ortho($P, $rootP)[0] + "
                   "rgbtohsl([$u, $v, $t])[0] + hsltorgb([$u, $v, $t])[2] + "
                   "saturate([$u, $v, $t], 0.3)[1] + hsi([$u, $v, $t], 30, 1.2, 0.8)[0] + "
                   "midhsi([$u, $v, $t], 30, 1.2, 0.8, 0.7, 1, 0)[2]",
                   D::Point, Agreement::Exact, 1.0);
    CompareFloat64(lane, "noise",
                   "noise($P) + snoise($P * 3) + vnoise($P)[1] + cnoise($P)[2] + "
                   "cellnoise($P * 5) + ccellnoise($P)[0] + pnoise($P, [4, 4, 4]) + "
                   "fbm($P, 4, 2.1, 0.55) + vfbm($P)[0] + cfbm($P)[1] + "
                   "turbulence($P) + vturbulence($P)[2] + cturbulence($P)[0]",
                   D::Point, Agreement::Exact, 1.0);
    CompareFloat64(lane, "voronoi-hash-rand",
                   "voronoi($P * 2) + voronoi($P * 2, 2) + voronoi($P * 2, 3) + "
                   "voronoi($P * 2, 4) + voronoi($P * 2, 5) + cvoronoi($P * 2)[1] + "
                   "pvoronoi($P * 2)[0] + hash($idLo, $pointIndex) + "
                   "rand() + rand(7) + rand(0, 1, 3)",
                   D::Point, Agreement::Exact, 1.0);
    CompareFloat64(lane, "curves",
                   "curve($t, 0, 1, 4, 0.5, 0.7, 4, 1, 0, 4) + "
                   "curve($t, 0, 0, 2, 0.5, 1, 1, 1, 0, 3) + "
                   "ccurve($t, 0, [1, 0, 0], 3, 1, [0, 1, 1], 3)[1] + "
                   "spline($t, 0, 0.3, 0.8, 1) + choose($t, 1, 2, 3) + "
                   "wchoose($t, 1, 2, 3, 4) + pick($t * 7, 1, 5, 2, 1, 3)",
                   D::Point, Agreement::Exact, 1.0);
    // Non-float-exact knot positions through interp 1 (linear) and 2
    // (smooth): both narrow the knots with NarrowToFloat, which a
    // double(float(x)) round trip fails to do on some drivers.
    CompareFloat64(lane, "curve-fknots-1",
                   "curve($t, 0.1, 1, 1, 0.35, 0.7, 1, 0.9, 0, 1)",
                   D::Point, Agreement::Exact, 1.0);
    CompareFloat64(lane, "curve-fknots-2",
                   "curve($t, 0.1, 1, 2, 0.35, 0.7, 2, 0.9, 0, 2)",
                   D::Point, Agreement::Exact, 1.0);
    CompareFloat64(lane, "transcendentals",
                   "sin($t) + cos($t) + tan($t * 0.5) + asin($t * 0.9) + acos($t * 0.9) + "
                   "atan($t) + sinh($t) + cosh($t) + tanh($t) + exp($t) + log($t + 1) + "
                   "log10($t + 1) + cbrt($t + 1) + pow($t + 1, 1.7) + atan2($u, $v + 1) + "
                   "angle($P, $rootP) + rotate($P, [0, 1, 0], $t)[0] + up($P, $rootP)[1] + "
                   "gamma($t + 0.1, 2.2) + bias($t * 0.8 + 0.1, 0.3) + contrast($t, 0.7) + "
                   "gaussstep($t, 0.2, 0.8)",
                   D::Point, Agreement::Transcendental, 1.0);
    CompareFloat64(lane, "locals-ifelse",
                   "$rootWidth = 1.0;\n"
                   "$tipWidth = 0.15;\n"
                   "$profile = curve($t, 0, 1, 4, 0.5, 0.7, 4, 1, 0, 4);\n"
                   "$jitter = 1;\n"
                   "if ($u > 0.5) { $jitter = 1 + 0.25 * rand(11); }\n"
                   "else { $jitter = 1 - 0.25 * rand(12); }\n"
                   "$value * mix($tipWidth, $rootWidth, $profile) * $jitter",
                   D::Point, Agreement::Exact, 0.012);
    CompareFloat64(
        lane, "groom", "$value * ($frame + 1)", D::Groom, Agreement::Exact, 3.0);
}

// Per-builtin matrix for the noise/voronoi/hash/rand family: one expression
// per builtin so a divergence names its function (kept from the original
// bisection because the aggregate above cannot).
void BisectNoiseVoronoi(Lane& lane) {
    using D = expr::Domain;
    CompareFloat64(lane, "b/noise", "noise($P)", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/snoise", "snoise($P * 3)", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/vnoise", "vnoise($P)[1]", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/cnoise", "cnoise($P)[2]", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/cellnoise", "cellnoise($P * 5)", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/ccellnoise", "ccellnoise($P)[0]", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/pnoise", "pnoise($P, [4, 4, 4])", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(lane, "b/fbm", "fbm($P, 4, 2.1, 0.55)", D::Point,
                   Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/vfbm", "vfbm($P)[0]", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/cfbm", "cfbm($P)[1]", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/turbulence", "turbulence($P)", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(lane, "b/vturbulence", "vturbulence($P)[2]", D::Point,
                   Agreement::Exact, 1.0);
    CompareFloat64(lane, "b/cturbulence", "cturbulence($P)[0]", D::Point,
                   Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/voronoi-def", "voronoi($P * 2)", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(lane, "b/voronoi-2", "voronoi($P * 2, 2)", D::Point,
                   Agreement::Exact, 1.0);
    CompareFloat64(lane, "b/voronoi-3", "voronoi($P * 2, 3)", D::Point,
                   Agreement::Exact, 1.0);
    CompareFloat64(lane, "b/voronoi-4", "voronoi($P * 2, 4)", D::Point,
                   Agreement::Exact, 1.0);
    CompareFloat64(lane, "b/voronoi-5", "voronoi($P * 2, 5)", D::Point,
                   Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/cvoronoi", "cvoronoi($P * 2)[1]", D::Point, Agreement::Exact, 1.0);
    // CVoronoi type 5: the second NarrowToFloat site (the default above is
    // type 1 and never reaches it).
    CompareFloat64(lane, "b/cvoronoi-5", "cvoronoi($P * 2, 5, 0.5)[0]", D::Point,
                   Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/pvoronoi", "pvoronoi($P * 2)[0]", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(lane, "b/hash2", "hash($idLo, $pointIndex)", D::Point,
                   Agreement::Exact, 1.0);
    CompareFloat64(lane, "b/rand0", "rand()", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(lane, "b/rand1", "rand(7)", D::Point, Agreement::Exact, 1.0);
    CompareFloat64(
        lane, "b/rand3", "rand(0, 1, 3)", D::Point, Agreement::Exact, 1.0);
}

// Every Store destination width, bitwise vs the CPU lane.
void StoreTypes(Lane& lane) {
    using D = expr::Domain;
    using T = expr::ScalarType;
    struct Case {
        char const* label;
        char const* source;
        T type;
        char const* native;
        VtValue literal;
    };
    Case const cases[] = {
        {"bool", "$t > 0.5", T::Bool, "bool", VtValue(true)},
        {"int32", "$pointIndex - 2*$primIndex", T::Int32, "int", VtValue(0)},
        {"uint32", "$pointIndex + $primIndex", T::UInt32, "uint", VtValue(0u)},
        {"int64", "$pointIndex * 1000000 - $primIndex", T::Int64, "int64",
         VtValue(int64_t(0))},
        {"uint64", "$pointIndex * 1000000 + $primIndex", T::UInt64, "uint64",
         VtValue(uint64_t(0))},
        {"float16", "$value * $u", T::Float16, "half", VtValue(GfHalf(0.5f))},
        {"float32", "$value * fit($t, 0, 1, 0.5, 1.5)", T::Float32, "float",
         VtValue(0.012f)},
        {"float64", "$value * $u", T::Float64, "double", VtValue(0.012)},
    };
    for (auto const& c : cases) {
        std::string const label = std::string("store/") + c.label;
        UsdGenGraphDesc graph =
            BindingGraph(c.source, c.native, c.type, 1, D::Point, c.literal);
        UsdGenExpressionBinding binding =
            MakeBinding(D::Point, c.literal, c.native, c.type, 1, "width");
        vulkan::ExprVkCompiledBinding compiled;
        std::vector<std::string> diagnostics;
        if (vulkan::ExprVkCompileBinding(usdGenUsd::Lower(graph), usdGenUsd::Lower(binding), true, true, &compiled,
                                         &diagnostics) !=
            vulkan::ExprVkCompileStatus::Ok) {
            Check(false, label + ": compiles");
            continue;
        }
        std::vector<vulkan::ExprVkCompiledBinding> bindings{compiled};
        if (lane.evaluator->Evaluate(bindings, lane.uploaded->geometry,
                                     lane.uploaded->channels, TestControls(D::Point),
                                     &diagnostics) !=
            vulkan::ExprVkParameterStatus::Ok) {
            Check(false, label + ": evaluates");
            continue;
        }
        auto const* field = lane.evaluator->Find(TfToken("width"));
        bool ok = false;
        std::vector<uint8_t> bytes = field && field->data
            ? Download(lane.context, *field->data, &ok)
            : std::vector<uint8_t>();
        if (!ok || bytes.size() != Scene::kPoints * 8) {
            Check(false, label + ": readback");
            continue;
        }
        // CPU oracle at the native width.
        expr::CpuExpressionContext built;
        expr::CpuCurveGeometryView view = lane.scene->Host();
        if (built.Build(view, TestControls(D::Point)) != expr::CpuExpressionStatus::Ok) {
            Check(false, label + ": cpu context");
            continue;
        }
        expr::FrontendOptions options;
        options.domain = D::Point;
        options.destination = c.type;
        options.components = 1;
        expr::CompileResult program = expr::Frontend::Compile(c.source, options);
        if (!program.ok) {
            Check(false, label + ": cpu compile");
            continue;
        }
        expr::IRProgram const ir = program.program.IR();
        size_t const widths[] = {0, 1, 4, 4, 8, 8, 2, 4, 8};
        size_t const width = widths[static_cast<unsigned>(c.type)];
        // The $value literal doubles, decoded per the binding's type.
        double literal = 0.0;
        if (c.literal.IsHolding<bool>()) literal = c.literal.UncheckedGet<bool>();
        else if (c.literal.IsHolding<int>()) literal = c.literal.UncheckedGet<int>();
        else if (c.literal.IsHolding<unsigned>())
            literal = c.literal.UncheckedGet<unsigned>();
        else if (c.literal.IsHolding<int64_t>())
            literal = double(c.literal.UncheckedGet<int64_t>());
        else if (c.literal.IsHolding<uint64_t>())
            literal = double(c.literal.UncheckedGet<uint64_t>());
        else if (c.literal.IsHolding<GfHalf>())
            literal = double(float(c.literal.UncheckedGet<GfHalf>()));
        else if (c.literal.IsHolding<float>())
            literal = c.literal.UncheckedGet<float>();
        else if (c.literal.IsHolding<double>())
            literal = c.literal.UncheckedGet<double>();
        expr::CpuExpressionInputs inputs = built.Inputs();
        expr::CpuExpressionField value;
        value.data = &literal;
        value.count = 1;
        value.domain = expr::Domain::Groom;
        value.components = 1;
        inputs.fields[static_cast<unsigned>(expr::Variable::Value)] = value;
        inputs.count = Scene::kPoints;
        std::vector<uint8_t> expect(Scene::kPoints * width, 0);
        expr::CpuExpressionOutput output;
        output.data = expect.data();
        output.count = Scene::kPoints;
        output.type = c.type;
        output.components = 1;
        if (expr::EvaluateProgramUnfused(ir, inputs, output) != expr::CpuExpressionStatus::Ok) {
            Check(false, label + ": cpu evaluate");
            continue;
        }
        bool bits = true;
        for (size_t i = 0; i < Scene::kPoints; ++i) {
            uint64_t slot = 0;
            std::memcpy(&slot, bytes.data() + i * 8, 8);
            uint64_t want = 0;
            std::memcpy(&want, expect.data() + i * width, width);
            if (slot != want) {
                if (bits)
                    std::printf("  %s: slot %zu vulkan %llx cpu %llx\n", label.c_str(),
                                i, (unsigned long long)slot, (unsigned long long)want);
                bits = false;
                if (i > 3) break;
            }
        }
        Check(bits, label + ": bitwise vs cpu");
    }
    // Float16 overflow fails closed (CUDA: Store rejects the infinity).
    {
        UsdGenGraphDesc graph = BindingGraph(
            "$cLength * 1000000", "half", expr::ScalarType::Float16, 1, D::Point,
            VtValue(GfHalf(0.5f)));
        UsdGenExpressionBinding binding = MakeBinding(
            D::Point, VtValue(GfHalf(0.5f)), "half", expr::ScalarType::Float16, 1, "width");
        vulkan::ExprVkCompiledBinding compiled;
        std::vector<std::string> diagnostics;
        bool const compiledOk =
            vulkan::ExprVkCompileBinding(usdGenUsd::Lower(graph), usdGenUsd::Lower(binding), true, true, &compiled,
                                         &diagnostics) == vulkan::ExprVkCompileStatus::Ok;
        bool invalid = false;
        if (compiledOk) {
            std::vector<vulkan::ExprVkCompiledBinding> bindings{compiled};
            invalid = lane.evaluator->Evaluate(bindings, lane.uploaded->geometry,
                                               lane.uploaded->channels,
                                               TestControls(D::Point),
                                               &diagnostics) ==
                vulkan::ExprVkParameterStatus::InvalidValue;
        }
        Check(compiledOk && invalid, "store/float16-overflow: InvalidValue");
    }
}

// Every materialized context variable, bitwise vs the CPU lane (point and
// primitive domains; with and without root frames).
void ContextParity(Lane& lane, vulkan::ExprVkContextPipeline* contextPipeline) {
    using V = expr::Variable;
    auto checkDomain = [&](expr::Domain domain, VkGeometry const& uploaded,
                           expr::CpuCurveGeometryView view, std::string const& tag) {
        size_t const n = domain == expr::Domain::Primitive ? Scene::kCurves
                                                           : Scene::kPoints;
        bool const frames = view.rootN != nullptr;
        vulkan::ExprVkFieldLayout layout = vulkan::ExprVkLayoutFields(
            domain, n, 1, 0, view.rest != nullptr, view.widths != nullptr,
            view.rootUV != nullptr, frames, frames, frames, view.rootPrim != nullptr);
        vulkan::ExprVkContextStatus status = vulkan::ExprVkContextStatus::DeviceError;
        auto built = contextPipeline->Begin(uploaded.geometry, uploaded.channels, domain,
                                            uint32_t(n), layout, &status);
        uint32_t semantic = UINT32_MAX;
        if (!built || built->Wait(30000000000ull, &semantic) != VK_SUCCESS ||
            semantic != 0) {
            Check(false, tag + ": builds");
            return;
        }
        bool ok = false;
        std::vector<uint8_t> bytes = Download(lane.context, *built->fields(), &ok);
        if (!ok || bytes.size() != size_t(layout.totalDoubles) * 8) {
            Check(false, tag + ": readback");
            return;
        }
        expr::CpuExpressionContext cpu;
        if (cpu.Build(view, TestControls(domain)) != expr::CpuExpressionStatus::Ok) {
            Check(false, tag + ": cpu context");
            return;
        }
        bool bits = true;
        for (unsigned v = 0; v < 33; ++v) {
            if (layout.offsets[v] == UINT64_MAX) continue;
            if (v == static_cast<unsigned>(V::Value)) continue; // evaluator-uploaded
            auto const& cpuField = cpu.Inputs().fields[v];
            if (!cpuField.data || cpuField.count * cpuField.components !=
                    size_t(layout.counts[v]) * layout.components[v]) {
                std::printf("  %s: var %u shape mismatch\n", tag.c_str(), v);
                bits = false;
                continue;
            }
            size_t const doubles = cpuField.count * cpuField.components;
            for (size_t i = 0; i < doubles; ++i) {
                uint64_t got = 0, want = 0;
                std::memcpy(&got, bytes.data() + (layout.offsets[v] + i) * 8, 8);
                std::memcpy(&want, cpuField.data + i, 8);
                if (got != want) {
                    if (bits) {
                        double g = 0;
                        std::memcpy(&g, &got, 8);
                        std::printf("  %s: var %u [%zu] vulkan %a cpu %a\n",
                                    tag.c_str(), v, i, g, cpuField.data[i]);
                    }
                    bits = false;
                    break;
                }
            }
        }
        // Owners map.
        bool ownersOk = false;
        std::vector<uint8_t> ownersBytes =
            Download(lane.context, *built->owners(), &ownersOk);
        auto const& cpuOwners = cpu.Inputs().pointToPrimitive;
        if (ownersOk && ownersBytes.size() == Scene::kPoints * 4 && cpuOwners.size) {
            for (size_t i = 0; i < Scene::kPoints; ++i) {
                uint32_t got = 0;
                std::memcpy(&got, ownersBytes.data() + i * 4, 4);
                if (got != cpuOwners.data[i]) {
                    ownersOk = false;
                    break;
                }
            }
        } else {
            ownersOk = false;
        }
        Check(bits, tag + ": fields bitwise vs cpu");
        Check(ownersOk, tag + ": owners vs cpu");
    };
    checkDomain(expr::Domain::Point, *lane.uploaded, lane.scene->Host(), "context/point");
    checkDomain(expr::Domain::Primitive, *lane.uploaded, lane.scene->Host(),
                "context/primitive");
    // Root frames + face ids.
    std::vector<float> rootN, rootT, rootB;
    std::vector<int32_t> rootPrim;
    for (unsigned c = 0; c < Scene::kCurves; ++c) {
        rootN.insert(rootN.end(), {0.0f, 1.0f, 0.0f});
        rootT.insert(rootT.end(), {1.0f, 0.0f, 0.0f});
        rootB.insert(rootB.end(), {0.0f, 0.0f, 1.0f});
        rootPrim.push_back(int32_t(c * 3 + 1));
    }
    VkGeometry framed = *lane.uploaded;
    framed.channels.rootN = Upload(lane.context, rootN.data(), rootN.size() * 4);
    framed.channels.rootT = Upload(lane.context, rootT.data(), rootT.size() * 4);
    framed.channels.rootB = Upload(lane.context, rootB.data(), rootB.size() * 4);
    framed.channels.rootPrim = Upload(lane.context, rootPrim.data(), rootPrim.size() * 4);
    if (!framed.channels.rootN || !framed.channels.rootT || !framed.channels.rootB ||
        !framed.channels.rootPrim) {
        Check(false, "context/frames: upload");
        return;
    }
    expr::CpuCurveGeometryView framedView = lane.scene->Host();
    framedView.rootN = rootN.data();
    framedView.rootT = rootT.data();
    framedView.rootB = rootB.data();
    framedView.rootPrim = rootPrim.data();
    checkDomain(expr::Domain::Point, framed, framedView, "context/frames");
}

// Expression-driven Width vs the same-order CPU formula (bitwise), plus the
// literal width.comp lane on constants when its .spv is provided.
void WidthApply(Lane& lane, VkNative& native,
                vulkan::ExprVkWidthApplyPipeline* widthApply,
                std::vector<uint32_t> const& literalWidthCode) {
    std::vector<float> input(Scene::kPoints);
    for (size_t i = 0; i < input.size(); ++i)
        input[i] = 0.012f + 0.001f * float(i % 6);
    std::vector<uint32_t> owners(Scene::kPoints);
    for (size_t i = 0; i < owners.size(); ++i) owners[i] = uint32_t(i / Scene::kCvs);
    auto inputBuffer = Upload(lane.context, input.data(), input.size() * 4);
    auto ownersBuffer = Upload(lane.context, owners.data(), owners.size() * 4);
    if (!inputBuffer || !ownersBuffer) {
        Check(false, "width: upload");
        return;
    }
    auto evaluateField = [&](std::string const& source, expr::Domain domain,
                             VtValue const& literal, std::string const& nativeType,
                             expr::ScalarType scalar, std::string const& destination,
                             std::shared_ptr<const vulkan::ChargedBuffer>* fieldOut) {
        UsdGenGraphDesc graph = BindingGraph(source, nativeType, scalar, 1, domain, literal);
        UsdGenExpressionBinding binding =
            MakeBinding(domain, literal, nativeType, scalar, 1, destination);
        vulkan::ExprVkCompiledBinding compiled;
        std::vector<std::string> diagnostics;
        if (vulkan::ExprVkCompileBinding(usdGenUsd::Lower(graph), usdGenUsd::Lower(binding), true, true, &compiled,
                                         &diagnostics) !=
            vulkan::ExprVkCompileStatus::Ok)
            return false;
        std::vector<vulkan::ExprVkCompiledBinding> bindings{compiled};
        if (lane.evaluator->Evaluate(bindings, lane.uploaded->geometry,
                                     lane.uploaded->channels, TestControls(domain),
                                     &diagnostics) !=
            vulkan::ExprVkParameterStatus::Ok)
            return false;
        auto const* field = lane.evaluator->Find(TfToken(destination));
        if (!field || !field->data) return false;
        *fieldOut = field->data;
        return true;
    };
    auto runWidth = [&](std::shared_ptr<const vulkan::ChargedBuffer> widthField,
                        uint32_t widthDom,
                        std::shared_ptr<const vulkan::ChargedBuffer> maskField,
                        uint32_t maskDom,
                        std::shared_ptr<const vulkan::ChargedBuffer> replaceField,
                        uint32_t replaceDom, float maskLiteral, uint32_t replaceLiteral,
                        std::vector<float>* out) {
        vulkan::ExprVkWidthApplyPipeline::Controls controls;
        controls.maskLiteral = maskLiteral;
        controls.widthDom = widthDom;
        controls.maskDom = maskDom;
        controls.replaceDom = replaceDom;
        controls.replaceLiteral = replaceLiteral;
        auto run = widthApply->Begin(inputBuffer, ownersBuffer, widthField, maskField,
                                     replaceField, nullptr, uint32_t(Scene::kPoints),
                                     uint32_t(Scene::kCurves), controls);
        uint32_t semantic = UINT32_MAX;
        if (!run || run->Wait(30000000000ull, &semantic) != VK_SUCCESS || semantic != 0)
            return false;
        bool ok = false;
        std::vector<uint8_t> bytes = DownloadDeviceLocal(native, *run->output(), &ok);
        if (!ok || bytes.size() != input.size() * 4) return false;
        out->assign(input.size(), 0.0f);
        std::memcpy(out->data(), bytes.data(), bytes.size());
        return true;
    };
    auto slotFloat = [&](std::shared_ptr<const vulkan::ChargedBuffer> const& field,
                         size_t slot) {
        bool ok = false;
        std::vector<uint8_t> bytes = Download(lane.context, *field, &ok);
        if (!ok) return 0.0f;
        uint32_t bits = 0;
        std::memcpy(&bits, bytes.data() + slot * 8, 4);
        float v = 0;
        std::memcpy(&v, &bits, 4);
        return v;
    };
    std::shared_ptr<const vulkan::ChargedBuffer> widthField;
    if (!evaluateField("$value * $u", expr::Domain::Point, VtValue(2.0f), "float",
                       expr::ScalarType::Float32, "width", &widthField)) {
        Check(false, "width: evaluates field");
        return;
    }
    // Replace with literal mask.
    {
        std::vector<float> out;
        bool const ran =
            runWidth(widthField, 4, nullptr, 0, nullptr, 0, 1.0f, 1, &out);
        bool bits = ran;
        for (size_t i = 0; bits && i < input.size(); ++i) {
            float const w = slotFloat(widthField, i);
            float const expect = input[i] + (w - input[i]);
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &out[i], 4);
            std::memcpy(&b, &expect, 4);
            bits = a == b;
        }
        Check(bits, "width/replace: bitwise vs formula");
    }
    // Multiply with literal mask.
    {
        std::vector<float> out;
        bool const ran =
            runWidth(widthField, 4, nullptr, 0, nullptr, 0, 1.0f, 0, &out);
        bool bits = ran;
        for (size_t i = 0; bits && i < input.size(); ++i) {
            float const w = slotFloat(widthField, i);
            float const expect = input[i] * (1.0f + (w - 1.0f));
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &out[i], 4);
            std::memcpy(&b, &expect, 4);
            bits = a == b;
        }
        Check(bits, "width/multiply: bitwise vs formula");
    }
    // Connected mask + connected per-curve replace.
    std::shared_ptr<const vulkan::ChargedBuffer> maskField, replaceField;
    if (!evaluateField("$t", expr::Domain::Point, VtValue(1.0f), "float",
                       expr::ScalarType::Float32, "mask", &maskField) ||
        !evaluateField("$primIndex > 1", expr::Domain::Primitive, VtValue(false), "bool",
                       expr::ScalarType::Bool, "replace", &replaceField)) {
        Check(false, "width: evaluates mask/replace");
        return;
    }
    {
        std::vector<float> out;
        bool const ran =
            runWidth(widthField, 4, maskField, 4, replaceField, 2, 1.0f, 0, &out);
        bool bits = ran;
        for (size_t i = 0; bits && i < input.size(); ++i) {
            float const w = slotFloat(widthField, i);
            float const m = slotFloat(maskField, i);
            bool ok = false;
            std::vector<uint8_t> rbytes = Download(lane.context, *replaceField, &ok);
            uint32_t raw = 0;
            if (ok) std::memcpy(&raw, rbytes.data() + (i / Scene::kCvs) * 8, 1);
            float const expect =
                (raw != 0) ? input[i] + (w - input[i]) * m
                           : input[i] * (1.0f + (w - 1.0f) * m);
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &out[i], 4);
            std::memcpy(&b, &expect, 4);
            bits = ok && a == b;
        }
        Check(bits, "width/connected: bitwise vs formula");
    }
    // Negative connected width fails closed (CUDA: BadValue).
    {
        std::shared_ptr<const vulkan::ChargedBuffer> bad;
        bool evaluated =
            evaluateField("$value - 1.0", expr::Domain::Point, VtValue(0.012f), "float",
                          expr::ScalarType::Float32, "width", &bad);
        vulkan::ExprVkWidthApplyPipeline::Controls controls;
        controls.widthDom = 4;
        auto run = evaluated ? widthApply->Begin(inputBuffer, ownersBuffer, bad, nullptr,
                                                 nullptr, nullptr,
                                                 uint32_t(Scene::kPoints),
                                                 uint32_t(Scene::kCurves), controls)
                             : nullptr;
        uint32_t semantic = 0;
        bool const badValue =
            evaluated && run && run->Wait(30000000000ull, &semantic) == VK_SUCCESS &&
            semantic == 3;
        Check(badValue, "width/negative: BadValue");
    }
    // Constant expressions agree with the literal lane bit-for-bit.
    if (!literalWidthCode.empty()) {
        auto literal = vulkan::WidthPipeline::Create(lane.context, literalWidthCode);
        std::shared_ptr<const vulkan::ChargedBuffer> constant;
        bool const evaluated =
            literal && evaluateField("$value", expr::Domain::Point, VtValue(0.5f),
                                     "float", expr::ScalarType::Float32, "width",
                                     &constant);
        std::vector<float> viaExpr;
        bool ranExpr = evaluated &&
            runWidth(constant, 4, nullptr, 0, nullptr, 0, 1.0f, 1, &viaExpr);
        auto literalRun = literal
            ? literal->Begin(inputBuffer, uint32_t(Scene::kPoints), 0.5f, 1)
            : nullptr;
        bool polled = false;
        std::vector<uint8_t> literalBytes;
        if (literalRun) {
            for (int spin = 0; spin < 30000; ++spin) {
                VkResult r = literalRun->Poll(nullptr);
                if (r == VK_SUCCESS) {
                    polled = true;
                    break;
                }
                if (r != VK_NOT_READY) break;
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            bool ok = false;
            if (polled && literalRun->succeeded())
                literalBytes = DownloadDeviceLocal(native, *literalRun->output(), &ok);
            polled = polled && ok && literalBytes.size() == input.size() * 4;
        }
        bool bits = ranExpr && polled;
        for (size_t i = 0; bits && i < input.size(); ++i) {
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &viaExpr[i], 4);
            std::memcpy(&b, literalBytes.data() + i * 4, 4);
            bits = a == b;
        }
        Check(bits, "width/literal-lane: bitwise vs width.comp");
    }
}

// UsdGenExprOp application vs the same-order CPU formula (bitwise).
void ExprOpApply(Lane& lane, VkNative& native,
                 vulkan::ExprVkExprOpPipeline* exprop) {
    std::vector<float> inPoints(Scene::kPoints * 3), inWidths(Scene::kPoints);
    for (size_t i = 0; i < Scene::kPoints; ++i) {
        inPoints[i * 3] = lane.scene->px[i];
        inPoints[i * 3 + 1] = lane.scene->py[i];
        inPoints[i * 3 + 2] = lane.scene->pz[i];
        inWidths[i] = 0.012f;
    }
    std::vector<uint32_t> owners(Scene::kPoints);
    for (size_t i = 0; i < owners.size(); ++i) owners[i] = uint32_t(i / Scene::kCvs);
    auto pointsBuffer = Upload(lane.context, inPoints.data(), inPoints.size() * 4);
    auto widthsBuffer = Upload(lane.context, inWidths.data(), inWidths.size() * 4);
    auto ownersBuffer = Upload(lane.context, owners.data(), owners.size() * 4);
    if (!pointsBuffer || !widthsBuffer || !ownersBuffer) {
        Check(false, "exprop: upload");
        return;
    }
    auto evaluate = [&](std::vector<vulkan::ExprVkCompiledBinding> bindings,
                        expr::Domain domain) {
        std::vector<std::string> diagnostics;
        return lane.evaluator->Evaluate(bindings, lane.uploaded->geometry,
                                        lane.uploaded->channels,
                                        TestControls(domain), &diagnostics) ==
            vulkan::ExprVkParameterStatus::Ok;
    };
    auto compile = [&](std::string const& source, expr::Domain domain, VtValue literal,
                       std::string const& nativeType, expr::ScalarType scalar,
                       unsigned components, std::string const& destination,
                       vulkan::ExprVkCompiledBinding* out) {
        UsdGenGraphDesc graph =
            BindingGraph(source, nativeType, scalar, components, domain, literal);
        UsdGenExpressionBinding binding =
            MakeBinding(domain, literal, nativeType, scalar, components, destination);
        return vulkan::ExprVkCompileBinding(usdGenUsd::Lower(graph), usdGenUsd::Lower(binding), true, true, out) ==
            vulkan::ExprVkCompileStatus::Ok;
    };
    auto runOp = [&](std::shared_ptr<const vulkan::ChargedBuffer> field,
                     std::shared_ptr<const vulkan::ChargedBuffer> maskField,
                     uint32_t maskDom, float maskLiteral, uint32_t returnType,
                     bool perCurve, std::vector<float>* outP,
                     std::vector<float>* outW) {
        auto run = exprop->Begin(pointsBuffer, widthsBuffer, ownersBuffer, field,
                                 maskField, uint32_t(Scene::kPoints),
                                 uint32_t(Scene::kCurves), returnType, perCurve,
                                 maskDom, maskLiteral);
        uint32_t semantic = UINT32_MAX;
        if (!run || run->Wait(30000000000ull, &semantic) != VK_SUCCESS || semantic != 0)
            return false;
        bool okP = false, okW = false;
        std::vector<uint8_t> bytesP = DownloadDeviceLocal(native, *run->outPoints(), &okP);
        std::vector<uint8_t> bytesW = DownloadDeviceLocal(native, *run->outWidths(), &okW);
        if (!okP || !okW || bytesP.size() != inPoints.size() * 4 ||
            bytesW.size() != inWidths.size() * 4)
            return false;
        outP->assign(inPoints.size(), 0.0f);
        outW->assign(inWidths.size(), 0.0f);
        std::memcpy(outP->data(), bytesP.data(), bytesP.size());
        std::memcpy(outW->data(), bytesW.data(), bytesW.size());
        return true;
    };
    auto slotFloats = [&](std::shared_ptr<const vulkan::ChargedBuffer> const& field,
                          size_t slot, unsigned count, std::vector<float>* out) {
        bool ok = false;
        std::vector<uint8_t> bytes = Download(lane.context, *field, &ok);
        if (!ok) return false;
        out->assign(count, 0.0f);
        for (unsigned k = 0; k < count; ++k) {
            uint32_t bits = 0;
            std::memcpy(&bits, bytes.data() + (slot + k) * 8, 4);
            std::memcpy(&(*out)[k], &bits, 4);
        }
        return true;
    };
    auto const clampMask = [](float m) { return m < 0.0f ? 0.0f : (1.0f < m ? 1.0f : m); };
    // Displacement, per-CV, literal mask (element 0 takes the exact-zero
    // bit-preserving path: $u=$v=$t=0 there).
    {
        vulkan::ExprVkCompiledBinding source;
        bool const compiledOk = compile("[$u, $v, $t]", expr::Domain::Point,
                                        VtValue(GfVec3f(0.0f)), "float3",
                                        expr::ScalarType::Float32, 3, "expr:source",
                                        &source);
        Check(compiledOk, "exprop/displacement-cv: compiles");
        if (!compiledOk) return;
        bool const evaluated = evaluate({source}, expr::Domain::Point);
        Check(evaluated, "exprop/displacement-cv: evaluates");
        if (!evaluated) return;
        auto const* field = lane.evaluator->Find(TfToken("expr:source"));
        std::vector<float> outP, outW;
        bool const ran = field && field->data &&
            runOp(field->data, nullptr, 0, 1.0f, 0, false, &outP, &outW);
        Check(ran, "exprop/displacement-cv: applies");
        if (!ran) return;
        bool bits = true;
        for (size_t i = 0; bits && i < Scene::kPoints; ++i) {
            std::vector<float> f;
            bits = slotFloats(field->data, i * 3, 3, &f);
            float const dx = f[0] * 1.0f, dy = f[1] * 1.0f, dz = f[2] * 1.0f;
            float ex, ey, ez;
            if (dx == 0.0f && dy == 0.0f && dz == 0.0f) {
                ex = inPoints[i * 3];
                ey = inPoints[i * 3 + 1];
                ez = inPoints[i * 3 + 2];
            } else {
                ex = inPoints[i * 3] + dx;
                ey = inPoints[i * 3 + 1] + dy;
                ez = inPoints[i * 3 + 2] + dz;
            }
            uint32_t a[3] = {0, 0, 0}, b[3] = {0, 0, 0};
            std::memcpy(a, &outP[i * 3], 12);
            float const ev[3] = {ex, ey, ez};
            std::memcpy(b, ev, 12);
            bits = a[0] == b[0] && a[1] == b[1] && a[2] == b[2];
            uint32_t wa = 0, wb = 0;
            std::memcpy(&wa, &outW[i], 4);
            std::memcpy(&wb, &inWidths[i], 4);
            bits = bits && wa == wb;
        }
        Check(bits, "exprop/displacement-cv: bitwise vs formula");
    }
    // Width, per-curve, connected mask (one Evaluate call runs each
    // binding in its own domain).
    {
        vulkan::ExprVkCompiledBinding source, mask;
        bool const ok =
            compile("$value + $u", expr::Domain::Primitive, VtValue(0.5f), "float",
                    expr::ScalarType::Float32, 1, "expr:source", &source) &&
            compile("$t", expr::Domain::Point, VtValue(1.0f), "float",
                    expr::ScalarType::Float32, 1, "mask", &mask) &&
            evaluate({source, mask}, expr::Domain::Primitive);
        auto const* field = lane.evaluator->Find(TfToken("expr:source"));
        auto const* maskField = lane.evaluator->Find(TfToken("mask"));
        std::vector<float> outP, outW;
        std::shared_ptr<const vulkan::ChargedBuffer> maskData =
            maskField ? maskField->data : nullptr;
        bool const ran = ok && field && field->data && maskData &&
            runOp(field->data, maskData, 4, 1.0f, 1, true, &outP, &outW);
        bool bits = ran;
        for (size_t i = 0; bits && i < Scene::kPoints; ++i) {
            uint32_t const c = uint32_t(i / Scene::kCvs);
            std::vector<float> f, m;
            bits = slotFloats(field->data, c, 1, &f) &&
                slotFloats(maskData, i, 1, &m);
            float const mc = clampMask(m[0]);
            float const raw = inWidths[i] + (f[0] - inWidths[i]) * mc;
            float const expect = 0.0f < raw ? raw : 0.0f;
            uint32_t a = 0, b = 0;
            std::memcpy(&a, &outW[i], 4);
            std::memcpy(&b, &expect, 4);
            bits = a == b;
            uint32_t pa[3] = {0, 0, 0}, pb[3] = {0, 0, 0};
            std::memcpy(pa, &outP[i * 3], 12);
            std::memcpy(pb, &inPoints[i * 3], 12);
            bits = bits && pa[0] == pb[0] && pa[1] == pb[1] && pa[2] == pb[2];
        }
        Check(bits, "exprop/width-curve-masked: bitwise vs formula");
    }
}

// Fail-closed paths: values, geometry, channels, and compile refusals.
void ErrorPaths(Lane& lane, vulkan::ExprVkContextPipeline* contextPipeline) {
    // Non-finite expression values never publish (CUDA: InvalidValue).
    {
        UsdGenGraphDesc graph = BindingGraph("$P[0] / ($P[0] - $P[0])", "double",
                                             expr::ScalarType::Float64, 1,
                                             expr::Domain::Point, VtValue(0.0));
        UsdGenExpressionBinding binding = MakeBinding(expr::Domain::Point,
                                                      VtValue(0.012), "double",
                                                      expr::ScalarType::Float64, 1,
                                                      "width");
        vulkan::ExprVkCompiledBinding compiled;
        bool const compiledOk =
            vulkan::ExprVkCompileBinding(usdGenUsd::Lower(graph), usdGenUsd::Lower(binding), true, true, &compiled) ==
            vulkan::ExprVkCompileStatus::Ok;
        bool invalid = false;
        if (compiledOk) {
            std::vector<vulkan::ExprVkCompiledBinding> bindings{compiled};
            invalid = lane.evaluator->Evaluate(bindings, lane.uploaded->geometry,
                                               lane.uploaded->channels,
                                               TestControls(expr::Domain::Point)) ==
                vulkan::ExprVkParameterStatus::InvalidValue;
        }
        Check(compiledOk && invalid, "error/inf: InvalidValue");
    }
    // SeExpr compile errors refuse at compile time (host-only).
    {
        UsdGenGraphDesc graph = BindingGraph("$value * ", "double",
                                             expr::ScalarType::Float64, 1,
                                             expr::Domain::Point, VtValue(0.0));
        UsdGenExpressionBinding binding = MakeBinding(expr::Domain::Point,
                                                      VtValue(0.012), "double",
                                                      expr::ScalarType::Float64, 1,
                                                      "width");
        vulkan::ExprVkCompiledBinding compiled;
        Check(vulkan::ExprVkCompileBinding(usdGenUsd::Lower(graph), usdGenUsd::Lower(binding), true, true, &compiled) ==
                vulkan::ExprVkCompileStatus::CompileError,
            "error/syntax: CompileError");
    }
    // Sampler programs refuse like the CUDA lane (host-only).
    {
        UsdGenGraphDesc graph = BindingGraph("geoSampler(\"g\", \"$index\")",
                                             "double", expr::ScalarType::Float64, 1,
                                             expr::Domain::Point, VtValue(0.0));
        UsdGenExpressionBinding binding = MakeBinding(expr::Domain::Point,
                                                      VtValue(0.012), "double",
                                                      expr::ScalarType::Float64, 1,
                                                      "width");
        vulkan::ExprVkCompiledBinding compiled;
        Check(vulkan::ExprVkCompileBinding(usdGenUsd::Lower(graph), usdGenUsd::Lower(binding), true, true, &compiled) ==
                vulkan::ExprVkCompileStatus::UnsupportedType,
            "error/sampler: UnsupportedType");
    }
    // ExprOp capture refusals (host-only).
    {
        UsdGenGraphDesc graph = BindingGraph("$value * 2", "float",
                                             expr::ScalarType::Float32, 1,
                                             expr::Domain::Point, VtValue(0.0f));
        UsdGenExpressionBinding binding = MakeBinding(expr::Domain::Point,
                                                      VtValue(0.012f), "float",
                                                      expr::ScalarType::Float32, 1,
                                                      "width");
        vulkan::ExprVkCompiledBinding compiled;
        Check(vulkan::ExprVkCompileBinding(usdGenUsd::Lower(graph), usdGenUsd::Lower(binding), false, true, &compiled) ==
                vulkan::ExprVkCompileStatus::CompileError,
            "error/exprop-value: CompileError");
        UsdGenGraphDesc timeGraph = BindingGraph("$frame + 1", "float",
                                                 expr::ScalarType::Float32, 1,
                                                 expr::Domain::Point, VtValue(0.0f));
        Check(vulkan::ExprVkCompileBinding(usdGenUsd::Lower(timeGraph), usdGenUsd::Lower(binding), true, false, &compiled) ==
                vulkan::ExprVkCompileStatus::CompileError,
            "error/exprop-time: CompileError");
    }
    // Device-side geometry/channel validation (CUDA: InvalidGeometry/Channel).
    {
        std::vector<uint32_t> badOffsets = lane.scene->offsets;
        badOffsets[1] = 99;
        auto badBuffer =
            Upload(lane.context, badOffsets.data(), badOffsets.size() * 4);
        VkGeometry bad = *lane.uploaded;
        bad.geometry.curveOffsets = badBuffer;
        vulkan::ExprVkFieldLayout layout = vulkan::ExprVkLayoutFields(
            expr::Domain::Point, Scene::kPoints, 1, 0, true, true, true, false,
            false, false, false);
        vulkan::ExprVkContextStatus status = vulkan::ExprVkContextStatus::Ok;
        auto built = badBuffer ? contextPipeline->Begin(bad.geometry, bad.channels,
                                                        expr::Domain::Point,
                                                        uint32_t(Scene::kPoints),
                                                        layout, &status)
                               : nullptr;
        uint32_t semantic = 0;
        bool const geometry = built &&
            built->Wait(30000000000ull, &semantic) == VK_SUCCESS && semantic == 1;
        Check(geometry, "error/offsets: InvalidGeometry");
    }
    {
        std::vector<float> badHairT = lane.scene->hairT;
        badHairT[0] = 2.0f;
        auto badBuffer = Upload(lane.context, badHairT.data(), badHairT.size() * 4);
        VkGeometry bad = *lane.uploaded;
        bad.channels.hairT = badBuffer;
        vulkan::ExprVkFieldLayout layout = vulkan::ExprVkLayoutFields(
            expr::Domain::Point, Scene::kPoints, 1, 0, true, true, true, false,
            false, false, false);
        vulkan::ExprVkContextStatus status = vulkan::ExprVkContextStatus::Ok;
        auto built = badBuffer ? contextPipeline->Begin(bad.geometry, bad.channels,
                                                        expr::Domain::Point,
                                                        uint32_t(Scene::kPoints),
                                                        layout, &status)
                               : nullptr;
        uint32_t semantic = 0;
        bool const channel = built &&
            built->Wait(30000000000ull, &semantic) == VK_SUCCESS && semantic == 2;
        Check(channel, "error/hairT: InvalidChannel");
    }
}

} // namespace

int main(int argc, char** argv) {
    std::vector<std::string> spv;
    bool forceLlvmpipe = false;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            forceLlvmpipe = std::string(argv[++i]) == "llvmpipe";
        } else {
            spv.push_back(arg);
        }
    }
    if (spv.size() != 4 && spv.size() != 5) {
        std::printf("usage: %s <evaluate.spv> <context.spv> <width.spv> <exprop.spv> "
                    "[literalWidth.spv] [--device nvidia|llvmpipe]\n",
                    argv[0]);
        return 2;
    }
    bool ok = false;
    std::vector<uint32_t> evaluateCode = LoadSpv(spv[0].c_str(), &ok);
    if (!ok) {
        std::printf("cannot load %s\n", spv[0].c_str());
        return 2;
    }
    std::vector<uint32_t> contextCode = LoadSpv(spv[1].c_str(), &ok);
    if (!ok) {
        std::printf("cannot load %s\n", spv[1].c_str());
        return 2;
    }
    std::vector<uint32_t> widthCode = LoadSpv(spv[2].c_str(), &ok);
    if (!ok) {
        std::printf("cannot load %s\n", spv[2].c_str());
        return 2;
    }
    std::vector<uint32_t> expropCode = LoadSpv(spv[3].c_str(), &ok);
    if (!ok) {
        std::printf("cannot load %s\n", spv[3].c_str());
        return 2;
    }
    std::vector<uint32_t> literalWidthCode;
    if (spv.size() == 5) {
        literalWidthCode = LoadSpv(spv[4].c_str(), &ok);
        if (!ok) {
            std::printf("cannot load %s\n", spv[4].c_str());
            return 2;
        }
    }
    bool nvidiaPending = false, unavailable = false;
    std::shared_ptr<VkNative> native = CreateDevice(forceLlvmpipe, &nvidiaPending,
                                                   &unavailable);
    if (!native) {
        if (unavailable) {
            std::printf("SKIP: no Vulkan device with float64+int64 compute\n");
            return 77;
        }
        std::printf("FAIL: device creation failed\n");
        return 1;
    }
    std::printf("device: %s%s\n", native->name.c_str(),
                native->isNvidia ? " (NVIDIA)" : " (secondary evidence)");
    vulkan::DeviceContext::CreateInfo info;
    info.instance = native->instance;
    info.physicalDevice = native->physical;
    info.device = native->device;
    info.computeQueue = native->queue;
    info.computeQueueFamily = native->family;
    info.physicalIndex = native->physicalIndex;
    info.resourceDeviceId = 0;
    info.nativeLifetime = native;
    info.gpuLabel = native->name;
    info.resources = {size_t{256} << 20, 0};
    info.shaderFloat64Enabled = true;
    auto context = vulkan::DeviceContext::Create(info);
    if (!context) {
        std::printf("FAIL: DeviceContext creation failed\n");
        return 1;
    }
    auto contextPipeline = vulkan::ExprVkContextPipeline::Create(context, contextCode);
    auto evaluatePipeline = vulkan::ExprVkEvaluatePipeline::Create(context, evaluateCode);
    auto widthApply = vulkan::ExprVkWidthApplyPipeline::Create(context, widthCode);
    auto exprop = vulkan::ExprVkExprOpPipeline::Create(context, expropCode);
    if (!contextPipeline || !evaluatePipeline || !widthApply || !exprop) {
        std::printf("FAIL: pipeline creation failed\n");
        return 1;
    }
    Scene scene;
    bool uploaded = false;
    VkGeometry geometry = UploadScene(context, scene, &uploaded);
    if (!uploaded) {
        std::printf("FAIL: scene upload failed\n");
        return 1;
    }
    vulkan::ExprVkParameterEvaluator evaluator(contextPipeline, evaluatePipeline);
#ifdef USDGEN_ENABLE_CUDA
    auto twin = SetupCudaTwin(scene);
    if (!twin->ready) std::printf("CUDA twin: %s\n", twin->note.c_str());
    if (!twin->ready && !twin->unavailable) {
        std::printf("FAIL: an available CUDA device could not initialize the parity twin\n");
        return 1;
    }
#endif
    Lane lane;
    lane.context = context;
    lane.evaluator = &evaluator;
    lane.uploaded = &geometry;
    lane.scene = &scene;
#ifdef USDGEN_ENABLE_CUDA
    lane.twin = twin.get();
#endif
    ExpressionMatrix(lane);
    BisectNoiseVoronoi(lane);
    StoreTypes(lane);
    ContextParity(lane, contextPipeline.get());
    WidthApply(lane, *native, widthApply.get(), literalWidthCode);
    ExprOpApply(lane, *native, exprop.get());
    ErrorPaths(lane, contextPipeline.get());
#ifdef USDGEN_ENABLE_CUDA
    bool const cudaParityUnavailable = !twin->ready;
    if (cudaParityUnavailable)
        std::printf("CUDA comparison unavailable (%s); Vulkan-only checks are secondary evidence.\n",
                    twin->note.c_str());
    else
        Check(g_cudaSkipped == 0, "every available CUDA twin comparison executed");
#else
    bool const cudaParityUnavailable = true;
    std::printf("CUDA twin not compiled; Vulkan-only checks are secondary evidence.\n");
#endif
    if (nvidiaPending) {
        std::printf("FAIL: available NVIDIA Vulkan device failed initialization; "
                    "fallback results are secondary evidence.\n");
        ++g_failures;
    }
    if (g_failures) {
        std::printf("RESULT: FAIL (%d)\n", g_failures);
        return 1;
    }
    if (cudaParityUnavailable) {
        std::printf("RESULT: SKIP (CUDA comparison unavailable)\n");
        return 77;
    }
    std::printf("RESULT: PASS\n");
    return 0;
}
