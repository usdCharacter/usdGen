// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Production default Vulkan Session provider.
//
// Owns a complete Vulkan device domain (instance, logical device, pipelines,
// completion service, plan executor) so plain Sessions route Vulkan without
// test-only injection. Device selection accepts discrete or integrated GPUs
// with Vulkan 1.2+ and timeline semaphores; software rasterizers (llvmpipe)
// and other non-GPU devices are rejected so a missing GPU reports as
// unavailable instead of silently executing on the CPU.
#include "defaultProvider.h"

#include "completionService.h"
#include "rbfVkDeformPipeline.h"
#include "resampleVkPipeline.h"
#include "deviceContext.h"
#include "deviceFactory.h"
#include "deviceGenerationAdapter.h"
#include "defaultResources.h"
#include "externalRetirement.h"
#include "executionPlan.h"
#include "exprVkContext.h"
#include "exprVkEvaluate.h"
#include "exprVkApply.h"
#include "exprVkPack.h"
#include "growVkPipeline.h"
#include "lengthCompactionPipeline.h"
#include "lengthScalePipeline.h"
#include "noisePipeline.h"
#include "nonWidthComparePipeline.h"
#include "picktileVk.h"
#include "planExecutor.h"
#include "sessionProvider.h"
#include "widthBlendPipeline.h"
#include "widthPipeline.h"
#include "usdGen/executionCache.h"
#include "usdGen/executionTaskGraph.h"
#include "usdGen/sessionDeviceProvider.h"
#include "pxr/base/arch/symbols.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <vector>

namespace usdGen::vulkan {
namespace {

void SetReason(std::string* reason, char const* message) noexcept {
    if (!reason) return;
    try { *reason = message; } catch (...) {}
}

// Raw Vulkan lifetime: instance, logical device and compute queue. Queued
// work must be idle before destruction; the wrapper Shutdown drains first.
struct DefaultNative {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    int physicalIndex = -1;
    ~DefaultNative() {
        if (device) vkDestroyDevice(device, nullptr);
        if (instance) vkDestroyInstance(instance, nullptr);
    }
};

bool LoadSpirvFile(char const* path, std::vector<uint32_t>* words) {
    if (!path || !*path || !words) return false;
    std::ifstream file(path, std::ios::binary);
    std::vector<char> raw((std::istreambuf_iterator<char>(file)), {});
    if (raw.empty() || raw.size() % sizeof(uint32_t) != 0) return false;
    words->resize(raw.size() / sizeof(uint32_t));
    std::memcpy(words->data(), raw.data(), raw.size());
    return true;
}

// Installed shaders live in share/usdGen/vulkan; build-tree shaders live in
// <build>/vulkan. USDGEN_VULKAN_SHADER_DIR overrides both (tests).
std::string ShaderDir() {
    if (char const* env = std::getenv("USDGEN_VULKAN_SHADER_DIR"))
        if (*env) return env;
    // Resolve beside this library so installed and relocated applications
    // do not depend on their working directory or the build workstation.
    static char locationAnchor;
    std::string library;
    if (pxr::ArchGetAddressInfo(&locationAnchor, &library, nullptr, nullptr, nullptr)) {
        auto const parent = std::filesystem::path(library).parent_path();
        for (auto const& dir : {parent / "vulkan",
                parent / USDGEN_VULKAN_SHADER_RELATIVE_DIR}) {
            std::ifstream file(dir / "width.spv", std::ios::binary);
            if (file.good()) return dir.lexically_normal().string();
        }
    }
    return {};
}

bool LoadShader(std::string const& dir, char const* name,
                std::vector<uint32_t>* words) {
    return LoadSpirvFile((dir + "/" + name).c_str(), words);
}

int DeviceScore(VkPhysicalDeviceType type) noexcept {
    switch (type) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
    case VK_PHYSICAL_DEVICE_TYPE_CPU: {
        // Opt in to software Vulkan for native API validation in CI.
        auto const* allow = std::getenv("USDGEN_VULKAN_ALLOW_SOFTWARE");
        return allow && std::strcmp(allow, "1") == 0 ? 1 : 0;
    }
    default: return 0;
    }
}

struct DevicePick {
    VkPhysicalDevice physical = VK_NULL_HANDLE;
    int index = -1;
    uint32_t family = UINT32_MAX;
    bool shaderFloat64 = false;
    bool shaderInt64 = false;
    bool memoryBudget = false;
    int score = 0;
};

bool HasMemoryBudget(VkPhysicalDevice physical) {
    uint32_t count = 0;
    if (vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, nullptr) != VK_SUCCESS)
        return false;
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateDeviceExtensionProperties(physical, nullptr, &count, extensions.data()) != VK_SUCCESS)
        return false;
    return std::any_of(extensions.begin(), extensions.end(), [](auto const& extension) {
        return std::strcmp(extension.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
    });
}

// Enumerates instance devices and picks the best GPU with a compute queue.
// Returns false when no suitable device exists (host has no GPU).
bool PickDevice(VkInstance instance, DevicePick* pick) noexcept {
    if (!instance || !pick) return false;
    try {
        uint32_t count = 0;
        if (vkEnumeratePhysicalDevices(instance, &count, nullptr) !=
                VK_SUCCESS ||
            !count)
            return false;
        std::vector<VkPhysicalDevice> devices(count);
        if (vkEnumeratePhysicalDevices(instance, &count, devices.data()) !=
            VK_SUCCESS)
            return false;
    DevicePick best;
    for (uint32_t i = 0; i < count; ++i) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.apiVersion < VK_API_VERSION_1_2) continue;
        int const score = DeviceScore(props.deviceType);
        if (!score || score < best.score) continue;
        VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
        timeline.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        VkPhysicalDeviceFeatures2 features{};
        features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features.pNext = &timeline;
        vkGetPhysicalDeviceFeatures2(devices[i], &features);
        if (!timeline.timelineSemaphore || !features.features.shaderInt64) continue;
        uint32_t families = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &families, nullptr);
        std::vector<VkQueueFamilyProperties> info(families);
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &families,
                                                 info.data());
        uint32_t chosen = UINT32_MAX;
        for (uint32_t f = 0; f < families; ++f) {
            if (!info[f].queueCount ||
                !(info[f].queueFlags & VK_QUEUE_COMPUTE_BIT))
                continue;
            // Prefer a dedicated compute queue over a graphics-capable one.
            if (chosen == UINT32_MAX ||
                ((info[chosen].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
                 !(info[f].queueFlags & VK_QUEUE_GRAPHICS_BIT)))
                chosen = f;
        }
        if (chosen == UINT32_MAX) continue;
        best.physical = devices[i];
        best.index = int(i);
        best.family = chosen;
        best.shaderFloat64 = features.features.shaderFloat64 == VK_TRUE;
        best.shaderInt64 = features.features.shaderInt64 == VK_TRUE;
        best.score = score;
    }
        if (!best.physical) return false;
        *pick = best;
        return true;
    } catch (...) {
        return false;
    }
}

std::shared_ptr<DefaultNative> CreateNative(DevicePick* pick,
                                            std::string* reason) {
    auto native = std::make_shared<DefaultNative>();
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "usdGen Vulkan default provider";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo instanceInfo{};
    instanceInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instanceInfo.pApplicationInfo = &app;
    VkResult const instanceResult =
        vkCreateInstance(&instanceInfo, nullptr, &native->instance);
    if (instanceResult != VK_SUCCESS) {
        auto const message = "Vulkan default provider cannot create an instance (VkResult " +
                             std::to_string(int(instanceResult)) + ")";
        SetReason(reason, message.c_str());
        return {};
    }
    // Select and query features/queues on the instance that owns the device.
    // Enumeration indices and handles from a destroyed probe are not reusable.
    if (!PickDevice(native->instance, pick)) {
        SetReason(reason, "no Vulkan compute device is available");
        return {};
    }
    native->physical = pick->physical;
    native->physicalIndex = pick->index;
    native->family = pick->family;
    pick->memoryBudget = HasMemoryBudget(native->physical);
    VkPhysicalDeviceTimelineSemaphoreFeatures timeline{};
    timeline.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    timeline.timelineSemaphore = VK_TRUE;
    VkPhysicalDeviceFeatures core{};
    core.shaderFloat64 = pick->shaderFloat64 ? VK_TRUE : VK_FALSE;
    core.shaderInt64 = pick->shaderInt64 ? VK_TRUE : VK_FALSE;
    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue{};
    queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue.queueFamilyIndex = native->family;
    queue.queueCount = 1;
    queue.pQueuePriorities = &priority;
    VkDeviceCreateInfo deviceInfo{};
    deviceInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceInfo.pNext = &timeline;
    deviceInfo.pEnabledFeatures = &core;
    deviceInfo.queueCreateInfoCount = 1;
    deviceInfo.pQueueCreateInfos = &queue;
    char const* memoryBudgetExtension = VK_EXT_MEMORY_BUDGET_EXTENSION_NAME;
    if (pick->memoryBudget) {
        deviceInfo.enabledExtensionCount = 1;
        deviceInfo.ppEnabledExtensionNames = &memoryBudgetExtension;
    }
    VkResult const deviceResult = vkCreateDevice(native->physical, &deviceInfo, nullptr,
                                                 &native->device);
    if (deviceResult != VK_SUCCESS) {
        auto const message = "Vulkan default provider cannot create a logical device (VkResult " +
                             std::to_string(int(deviceResult)) + ")";
        SetReason(reason, message.c_str());
        return {};
    }
    vkGetDeviceQueue(native->device, native->family, 0, &native->queue);
    if (!native->queue) {
        SetReason(reason, "Vulkan default provider has no compute queue");
        return {};
    }
    return native;
}

// Delegating wrapper: holds every native lifetime (instance/device, runtime,
// pipelines, executor) behind the neutral provider interface.
class DefaultVulkanSessionProvider final
    : public UsdGenSessionDeviceProvider,
      public std::enable_shared_from_this<DefaultVulkanSessionProvider> {
public:
    static std::shared_ptr<DefaultVulkanSessionProvider> Create(
        std::string* reason);
    ~DefaultVulkanSessionProvider() override { Shutdown(); }

    UsdGenSessionDevicePlanHandle Compile(
        UsdGenGraphDesc const& desc,
        UsdGenDiagnostics* diagnostics) const override {
        return inner_ ? inner_->Compile(desc, diagnostics) : nullptr;
    }
    UsdGenSessionDeviceIdentity Identity() const noexcept override {
        return inner_ ? inner_->Identity() : identity_;
    }
    bool NotifyContextLost(
        UsdGenSessionDeviceIdentity const& lost) noexcept override {
        bool const accepted = inner_ && inner_->NotifyContextLost(lost);
        if (accepted && domain_) domain_->StopAdmission();
        return accepted;
    }
    bool Submit(UsdGenSessionDeviceRequest request) override {
        return inner_ ? inner_->Submit(std::move(request)) : false;
    }
    void Shutdown() noexcept override {
        bool expected = false;
        if (!closing_.compare_exchange_strong(expected, true,
                                             std::memory_order_acq_rel))
            return;
        if (domain_) domain_->StopAdmission();
        identity_ = Identity();
        // A later provider can initialize driver modules after the queue's
        // first-use exit hook. Refresh once at this provider's final boundary,
        // after all lazy pipeline work, before deferring native destruction.
        bool const exitRegistered = RegisterVulkanExternalRetirementAtExit();
        // Join/drain are external ownership boundaries. Destruction may be
        // triggered by the last Session completion on a pipeline callback;
        // move the full native domain to the tracked external close queue.
        struct Retirement : std::enable_shared_from_this<Retirement> {
            std::shared_ptr<DefaultNative> native;
            std::shared_ptr<UsdGenExecutionRuntime> runtime;
            std::shared_ptr<UsdGenExecutionPipeline> queueOwner;
            std::shared_ptr<VulkanCompletionService> completion;
            std::shared_ptr<VulkanGenerationAdapterDomain> domain;
            std::shared_ptr<UsdGenExecutionTaskGraph> preparer;
            std::shared_ptr<DeviceContext> context;
            std::shared_ptr<VulkanPlanExecutor> executor;
            std::shared_ptr<VulkanSessionProvider> inner;
            std::shared_ptr<VulkanExternalRetirementDomain> exitDomain;
            bool exitRegistered = false;
            void Close() noexcept {
                try {
                    if (inner) inner->Shutdown();
                    if (executor) executor->Shutdown();
                    if (!exitRegistered) {
                        // Registration failure (or already-closed admission)
                        // cannot authorize deferred destruction during driver
                        // teardown. Stop its waiter externally, then retain
                        // every native capture instead of running destructors.
                        if (exitDomain) exitDomain->Quiesce();
                        else {
                            if (queueOwner) queueOwner->Shutdown();
                            if (completion) completion->CloseAndJoin();
                        }
                        auto* retained = new std::shared_ptr<Retirement>(shared_from_this());
                        (void)retained;
                        return;
                    }
                    auto lifecycleLock = exitDomain ? exitDomain->Lock()
                        : std::unique_lock<std::mutex>{};
                    if (exitDomain && exitDomain->IsQuiesced()) {
                        auto* retained = new std::shared_ptr<Retirement>(shared_from_this());
                        (void)retained;
                        return;
                    }
                    if (preparer) preparer->Drain();
                    if (queueOwner) queueOwner->Drain();
                    if (domain && !domain->Close()) {
                        // The last consumer proof schedules a new external
                        // close; no owner callback waits and no healthy lease
                        // forces permanent retention of the provider domain.
                        auto held = shared_from_this();
                        if (!domain->RetireWhenIdle([held] { held->Close(); })) {
                            auto* retained = new std::shared_ptr<Retirement>(std::move(held));
                            (void)retained;
                        }
                        return;
                    }
                    if (completion && !completion->CloseAndJoin()) {
                        auto* retained = new std::shared_ptr<Retirement>(shared_from_this());
                        (void)retained;
                    }
                } catch (...) {
                    auto* retained = new std::shared_ptr<Retirement>(shared_from_this());
                    (void)retained;
                }
            }
        };
        try {
            auto retirement = std::make_shared<Retirement>();
            retirement->native = std::move(native_);
            retirement->runtime = std::move(runtime_);
            retirement->queueOwner = std::move(queueOwner_);
            retirement->completion = std::move(completion_);
            retirement->domain = std::move(domain_);
            retirement->preparer = std::move(preparer_);
            retirement->context = std::move(context_);
            retirement->executor = std::move(executor_);
            retirement->inner = std::move(inner_);
            retirement->exitDomain = std::move(exitDomain_);
            retirement->exitRegistered = exitRegistered;
            if (!UsdGenExecutionPipeline::IsExecuting()) retirement->Close();
            else {
                try {
                    if (!EnqueueVulkanExternalRetirement([retirement] { retirement->Close(); })) {
                        auto* retained = new std::shared_ptr<Retirement>(std::move(retirement));
                        (void)retained;
                    }
                }
                catch (...) {
                    auto* retained = new std::shared_ptr<Retirement>(std::move(retirement));
                    (void)retained;
                }
            }
        } catch (...) {
            // Allocation failure cannot justify dropping an active waiter.
            std::terminate();
        }
    }

private:
    DefaultVulkanSessionProvider() = default;
    std::shared_ptr<DefaultNative> native_;
    std::shared_ptr<UsdGenExecutionRuntime> runtime_;
    std::shared_ptr<UsdGenExecutionPipeline> queueOwner_;
    std::shared_ptr<VulkanCompletionService> completion_;
    std::shared_ptr<VulkanGenerationAdapterDomain> domain_;
    std::shared_ptr<UsdGenExecutionTaskGraph> preparer_;
    std::shared_ptr<DeviceContext> context_;
    std::shared_ptr<VulkanPlanExecutor> executor_;
    std::shared_ptr<VulkanSessionProvider> inner_;
    std::shared_ptr<VulkanExternalRetirementDomain> exitDomain_;
    UsdGenSessionDeviceIdentity identity_{};
    std::atomic<bool> closing_{false};
};

std::shared_ptr<DefaultVulkanSessionProvider>
DefaultVulkanSessionProvider::Create(std::string* reason) {
    DevicePick pick;
    // Load shaders before touching the device: a missing width module fails
    // with no Vulkan state at all (safe on any thread), and optional
    // modules only narrow later pipeline admission.
    std::string const dir = ShaderDir();
    std::vector<uint32_t> widthSpv;
    if (!LoadShader(dir, "width.spv", &widthSpv)) {
        SetReason(reason, "Vulkan default provider cannot load width.spv");
        return {};
    }
    std::vector<uint32_t> scale, set, cut, reparam, minimum, literal, envelope, lengthDevice;
    bool const hasScale = LoadShader(dir, "lengthScale.spv", &scale);
    bool const hasSet = LoadShader(dir, "lengthSet.spv", &set);
    bool const hasCut = LoadShader(dir, "lengthCutExtend.spv", &cut);
    bool const hasReparam = LoadShader(dir, "lengthReparam.spv", &reparam);
    bool const hasMinimum = LoadShader(dir, "lengthMinimum.spv", &minimum);
    bool const hasLiteral = LoadShader(dir, "lengthLiteralV1.spv", &literal);
    bool const hasEnvelope = LoadShader(dir, "lengthEnvelopeV1.spv", &envelope);
    bool const hasLengthDevice = LoadShader(dir, "lengthDeviceV1.spv", &lengthDevice);
    std::vector<uint32_t> blendSpv, compareSpv, compactSpv;
    bool const hasBlend = LoadShader(dir, "widthBlend.spv", &blendSpv);
    bool const hasCompare = LoadShader(dir, "nonWidthCompare.spv", &compareSpv);
    bool const hasCompact = LoadShader(dir, "lengthCompaction.spv", &compactSpv);
    std::vector<uint32_t> noiseValidateSpv, noiseGenerateSpv;
    bool const hasNoise =
        LoadShader(dir, "noiseValidate.spv", &noiseValidateSpv) &&
        LoadShader(dir, "noiseGenerate.spv", &noiseGenerateSpv);
    std::vector<uint32_t> resampleSpv;
    bool const hasResample = LoadShader(dir, "resampleVk.spv", &resampleSpv);
    std::vector<uint32_t> exprContextSpv, exprEvaluateSpv, exprWidthSpv, exprPackSpv, growSpv;
    bool const hasExprContext = LoadShader(dir, "exprVkContext.spv", &exprContextSpv);
    bool const hasExprEvaluate = LoadShader(dir, "exprVkEvaluate.spv", &exprEvaluateSpv);
    bool const hasExprWidth = LoadShader(dir, "exprVkWidth.spv", &exprWidthSpv);
    bool const hasExprPack = LoadShader(dir, "exprVkPack.spv", &exprPackSpv);
    bool const hasGrow = LoadShader(dir, "curveGrow.spv", &growSpv);
    // Phase 2 picktiles: optional tile-span/bounds finalization; missing
    // shaders only disable the tile stage (publications carry no tiles).
    std::vector<uint32_t> tilesSpv, boundsSpv;
    bool const hasTiles = LoadShader(dir, "picktileVkTiles.spv", &tilesSpv) &&
        LoadShader(dir, "picktileVkBounds.spv", &boundsSpv);
    // Phase-2 RBF device path (rbfVk hook): optional modules; missing
    // shaders narrow all Deform admission (see planExecutor.cpp).
    std::vector<uint32_t> rbfVkExtentSpv, rbfVkGramSpv, rbfVkBuildMatrixSpv, rbfVkLuSpv,
        rbfVkRhsSpv, rbfVkTriSolveSpv, rbfVkEvaluateSpv, rbfVkApplySpv;
    bool const hasRbfVk =
        LoadShader(dir, "rbfVkExtent.spv", &rbfVkExtentSpv) &&
        LoadShader(dir, "rbfVkGram.spv", &rbfVkGramSpv) &&
        LoadShader(dir, "rbfVkBuildMatrix.spv", &rbfVkBuildMatrixSpv) &&
        LoadShader(dir, "rbfVkLu.spv", &rbfVkLuSpv) &&
        LoadShader(dir, "rbfVkRhs.spv", &rbfVkRhsSpv) &&
        LoadShader(dir, "rbfVkTriSolve.spv", &rbfVkTriSolveSpv) &&
        LoadShader(dir, "rbfVkEvaluate.spv", &rbfVkEvaluateSpv) &&
        LoadShader(dir, "rbfVkApply.spv", &rbfVkApplySpv);
    // R-cache programs: optional, never gating. Missing shaders only
    // disable the RbfVkBinding evaluate cache (direct evaluation, as
    // before); the binding rejects a partial set at Create.
    std::vector<uint32_t> rbfVkVerifySpv, rbfVkFillSpv, rbfVkEvaluateCachedSpv;
    bool const hasRbfVkEvalCache =
        LoadShader(dir, "rbfVkVerify.spv", &rbfVkVerifySpv) &&
        LoadShader(dir, "rbfVkFill.spv", &rbfVkFillSpv) &&
        LoadShader(dir, "rbfVkEvaluateCached.spv", &rbfVkEvaluateCachedSpv);
    if (!hasRbfVkEvalCache) {
        rbfVkVerifySpv.clear();
        rbfVkFillSpv.clear();
        rbfVkEvaluateCachedSpv.clear();
    }
    auto self = std::shared_ptr<DefaultVulkanSessionProvider>(
        new DefaultVulkanSessionProvider);
    self->native_ = CreateNative(&pick, reason);
    if (!self->native_) return {};
    struct Lifetime { std::shared_ptr<DefaultNative> native; };
    auto lifetime = std::make_shared<Lifetime>();
    lifetime->native = self->native_;
    self->runtime_ = std::make_shared<UsdGenExecutionRuntime>(4);
    self->queueOwner_ = std::make_shared<UsdGenExecutionPipeline>(
        *self->runtime_, 64, 64);
    VkPhysicalDeviceIDProperties ids{};
    ids.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES;
    VkPhysicalDeviceProperties2 properties{};
    properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    properties.pNext = &ids;
    vkGetPhysicalDeviceProperties2(self->native_->physical, &properties);
    std::array<uint8_t, VK_UUID_SIZE> uuid{};
    std::copy(std::begin(ids.deviceUUID), std::end(ids.deviceUUID), uuid.begin());
    std::shared_ptr<const DeviceFactory::Entry> entry;
    auto replyLifetime = std::make_shared<int>(0);
    VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
    budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
    VkPhysicalDeviceMemoryProperties2 memory{};
    memory.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memory.pNext = pick.memoryBudget ? &budget : nullptr;
    vkGetPhysicalDeviceMemoryProperties2(self->native_->physical, &memory);
    auto const resources = VulkanDefaultResourceConfig(
        memory.memoryProperties, pick.memoryBudget ? &budget : nullptr);
    // Construction runs on an external bring-up thread. Registry resolution
    // carries only UUID/configuration state and never owns native objects.
    self->queueOwner_->Await([&](auto done) {
        if (!DeviceFactory::ResolveAsync(uuid, resources,
                *self->queueOwner_, replyLifetime,
                [&, done](auto resolved, auto outcome) {
                    if (outcome == DeviceFactory::ResolveOutcome::Resolved)
                        entry = std::move(resolved);
                    done();
                }, false)) done();
    });
    if (!entry) {
        SetReason(reason, "Vulkan default provider resource identity resolution failed");
        return {};
    }
    DeviceFactory::CreateInfo contextInfo;
    contextInfo.instance = self->native_->instance;
    contextInfo.physicalDevice = self->native_->physical;
    contextInfo.device = self->native_->device;
    contextInfo.computeQueue = self->native_->queue;
    contextInfo.computeQueueFamily = self->native_->family;
    contextInfo.physicalIndex = self->native_->physicalIndex;
    contextInfo.nativeLifetime = lifetime;
    contextInfo.gpuLabel = "default";
    contextInfo.timelineSemaphoreEnabled = true;
    contextInfo.shaderFloat64Enabled = pick.shaderFloat64;
    contextInfo.shaderInt64Enabled = pick.shaderInt64;
    VkResult status = VK_SUCCESS;
    self->context_ = DeviceFactory::CreateContext(entry, contextInfo, &status);
    if (!self->context_ || status != VK_SUCCESS) {
        SetReason(reason, "Vulkan default provider cannot create a device context");
        return {};
    }
    // Pipelines need only the device context; build them before the
    // completion waiter exists so a pipeline failure destroys plain Vk
    // objects inline (safe on any thread) instead of needing the reaper.
    auto width = WidthPipeline::Create(self->context_, widthSpv, &status);
    if (!width) {
        SetReason(reason, "Vulkan default provider cannot create the width pipeline");
        return {};
    }
    // Length scale/set/cut/reparam/minimum/literal/envelope chain: install
    // the largest contiguous combination present; missing shaders only
    // narrow admission for graphs needing those Length forms.
    std::shared_ptr<LengthScalePipeline> length;
    if (hasScale && hasSet && hasCut && hasReparam && hasMinimum && hasLiteral &&
        hasEnvelope && hasLengthDevice)
        length = LengthScalePipeline::CreateWithDeviceV1(
            self->context_, scale, set, cut, reparam, minimum, literal,
            envelope, lengthDevice);
    else if (hasScale && hasSet && hasCut && hasReparam && hasMinimum && hasLiteral &&
        hasEnvelope)
        length = LengthScalePipeline::CreateWithEnvelopeV1(
            self->context_, scale, set, cut, reparam, minimum, literal,
            envelope);
    else if (hasScale && hasSet && hasCut && hasReparam && hasMinimum &&
             hasLiteral)
        length = LengthScalePipeline::CreateWithLiteralV1(
            self->context_, scale, set, cut, reparam, minimum, literal);
    else if (hasScale && hasSet && hasCut && hasReparam && hasMinimum)
        length = LengthScalePipeline::CreateWithMinimum(
            self->context_, scale, set, cut, reparam, minimum);
    else if (hasScale && hasSet && hasCut && hasReparam)
        length = LengthScalePipeline::CreateWithReparam(self->context_, scale,
                                                        set, cut, reparam);
    else if (hasScale && hasSet && hasCut)
        length = LengthScalePipeline::CreateWithCutExtend(self->context_, scale,
                                                          set, cut);
    else if (hasScale && hasSet)
        length = LengthScalePipeline::CreateWithSet(self->context_, scale, set);
    else if (hasScale)
        length = LengthScalePipeline::Create(self->context_, scale);
    std::shared_ptr<WidthBlendPipeline> blend;
    if (hasBlend) blend = WidthBlendPipeline::Create(self->context_, blendSpv);
    std::shared_ptr<NonWidthComparePipeline> compare;
    if (hasCompare)
        compare = NonWidthComparePipeline::Create(self->context_, compareSpv);
    std::shared_ptr<LengthCompactionPipeline> compact;
    if (hasCompact)
        compact = LengthCompactionPipeline::CreateWithKeep(self->context_, compactSpv);
    std::shared_ptr<NoisePipeline> noise;
    if (pick.shaderFloat64 && hasNoise)
        noise = NoisePipeline::Create(self->context_, noiseValidateSpv,
                                      noiseGenerateSpv);
    std::shared_ptr<ResampleVkPipeline> resample;
    if (pick.shaderFloat64 && hasResample)
        resample = ResampleVkPipeline::Create(self->context_, resampleSpv);
    std::shared_ptr<ExprVkContextPipeline> expressionContext;
    std::shared_ptr<ExprVkEvaluatePipeline> expressionEvaluate;
    std::shared_ptr<ExprVkWidthApplyPipeline> expressionWidth;
    std::shared_ptr<ExprVkPackPipeline> expressionPack;
    if (pick.shaderFloat64 && pick.shaderInt64 && hasExprContext && hasExprEvaluate) {
        expressionContext = ExprVkContextPipeline::Create(self->context_, exprContextSpv);
        expressionEvaluate = ExprVkEvaluatePipeline::Create(self->context_, exprEvaluateSpv);
    }
    if (hasExprWidth)
        expressionWidth = ExprVkWidthApplyPipeline::Create(self->context_, exprWidthSpv);
    if (hasExprPack)
        expressionPack = ExprVkPackPipeline::Create(self->context_, exprPackSpv);
    std::shared_ptr<GrowVkPipeline> grow;
    if (hasGrow && resample) {
        auto nativeGrow = CurveGrowPipeline::Create(self->context_, growSpv);
        if (nativeGrow) grow = GrowVkPipeline::Create(nativeGrow, resample);
    }
    std::shared_ptr<PicktileVkTilesPipeline> tiles;
    std::shared_ptr<PicktileVkBoundsPipeline> bounds;
    if (hasTiles) {
        tiles = PicktileVkTilesPipeline::Create(self->context_, tilesSpv);
        bounds = PicktileVkBoundsPipeline::Create(self->context_, boundsSpv);
        if (!tiles || !bounds) { tiles.reset(); bounds.reset(); }
    }
    std::shared_ptr<RbfVkDeformPipeline> rbfVkDeform;
    if (pick.shaderFloat64 && hasRbfVk) {
        RbfVkBindingSpirv rbfVkSpirv;
        rbfVkSpirv.extent = std::move(rbfVkExtentSpv);
        rbfVkSpirv.gram = std::move(rbfVkGramSpv);
        rbfVkSpirv.buildMatrix = std::move(rbfVkBuildMatrixSpv);
        rbfVkSpirv.lu = std::move(rbfVkLuSpv);
        rbfVkSpirv.rhs = std::move(rbfVkRhsSpv);
        rbfVkSpirv.triSolve = std::move(rbfVkTriSolveSpv);
        rbfVkSpirv.evaluate = std::move(rbfVkEvaluateSpv);
        rbfVkSpirv.verify = std::move(rbfVkVerifySpv);
        rbfVkSpirv.fill = std::move(rbfVkFillSpv);
        rbfVkSpirv.evaluateCached = std::move(rbfVkEvaluateCachedSpv);
        rbfVkDeform = RbfVkDeformPipeline::Create(self->context_, rbfVkSpirv, rbfVkApplySpv);
    }
    self->completion_ = VulkanCompletionService::Create(
        {self->context_, self->queueOwner_.get(), 64});
    if (!self->completion_) {
        SetReason(reason, "Vulkan default provider cannot create a completion service");
        return {};
    }
    // From here on a running completion waiter exists. Shutdown owns its
    // external join boundary, including tracked retirement if called from
    // graph work; partial construction must use that same lifetime path.
    auto Fail = [&](char const* message)
        -> std::shared_ptr<DefaultVulkanSessionProvider> {
        SetReason(reason, message);
        self->Shutdown();
        return {};
    };
    self->exitDomain_ = TrackVulkanExternalRetirementDomain(self->queueOwner_, self->completion_);
    if (!self->exitDomain_)
        return Fail("Vulkan default provider cannot reserve process retirement tracking");
    self->domain_ = VulkanGenerationAdapterDomain::Create(
        {self->queueOwner_, self->completion_});
    if (!self->domain_)
        return Fail("Vulkan default provider cannot create a generation domain");
    self->preparer_ = UsdGenExecutionTaskGraph::GetOrCreate(
        *self->runtime_, "vulkan", self->context_->resourceDeviceId());
    std::string executorReason;
    VulkanPlanExecutor::CreateInfo executorInfo;
    executorInfo.domain = self->domain_;
    executorInfo.widthPipeline = width;
    executorInfo.preparer = self->preparer_;
    executorInfo.lengthPipeline = length;
    executorInfo.widthBlendPipeline = blend;
    executorInfo.nonWidthComparePipeline = compare;
    executorInfo.lengthCompactionPipeline = compact;
    executorInfo.noisePipeline = noise;
    // Production deformation requires the device solver. Missing FP64 or
    // RBF modules fail admission; legacy host-fit pipelines remain available
    // only to callers that explicitly inject a compatibility provider.
    executorInfo.resampleVkPipeline = resample;
    executorInfo.tilesPipeline = tiles;
    executorInfo.boundsPipeline = bounds;
    executorInfo.rbfVkDeformPipeline = rbfVkDeform;
    executorInfo.expressionContextPipeline = expressionContext;
    executorInfo.expressionEvaluatePipeline = expressionEvaluate;
    executorInfo.expressionWidthPipeline = expressionWidth;
    executorInfo.expressionPackPipeline = expressionPack;
    executorInfo.growVkPipeline = grow;
    self->executor_ = VulkanPlanExecutor::Create(std::move(executorInfo), &executorReason);
    if (!self->executor_) {
        std::string message = executorReason.empty()
            ? "Vulkan default provider cannot create a plan executor"
            : executorReason;
        return Fail(message.c_str());
    }
    UsdGenSessionDeviceIdentity identity;
    identity.backend = UsdGenDeviceBackend::Vulkan;
    identity.deviceIndex = self->context_->physicalIndex();
    identity.capabilityVersion = 1;
    identity.logicalContextIdentity =
        reinterpret_cast<uintptr_t>(self->context_.get());
    identity.contextEpoch = 0;
    std::copy(self->context_->deviceUUID().begin(),
              self->context_->deviceUUID().end(),
              identity.physicalDeviceUuid.begin());
    self->identity_ = identity;
    std::string providerReason;
    self->inner_ = VulkanSessionProvider::Create(
        {self->executor_, self->context_, self->queueOwner_, identity, 256},
        &providerReason);
    if (!self->inner_) {
        std::string message = providerReason.empty()
            ? "Vulkan default provider cannot create a session provider"
            : providerReason;
        return Fail(message.c_str());
    }
    if (!RegisterVulkanExternalRetirementAtExit())
        return Fail("Vulkan default provider cannot register its process shutdown boundary");
    return self;
}

} // namespace

std::shared_ptr<UsdGenSessionDeviceProvider> CreateDefaultVulkanSessionProvider(
    std::string* reason) {
    if (UsdGenExecutionPipeline::IsExecuting()) {
        SetReason(reason, "Vulkan provider creation requires an external boundary");
        return {};
    }
    if (reason) reason->clear();
    try { return DefaultVulkanSessionProvider::Create(reason); }
    catch (...) {
        SetReason(reason, "Vulkan default provider construction failed");
        return {};
    }
}

bool HasDefaultVulkanDevice() noexcept {
    VkInstance probe = VK_NULL_HANDLE;
    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "usdGen Vulkan default probe";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    info.pApplicationInfo = &app;
    if (vkCreateInstance(&info, nullptr, &probe) != VK_SUCCESS) return false;
    DevicePick pick;
    bool const found = PickDevice(probe, &pick);
    vkDestroyInstance(probe, nullptr);
    return found;
}

} // namespace usdGen::vulkan
