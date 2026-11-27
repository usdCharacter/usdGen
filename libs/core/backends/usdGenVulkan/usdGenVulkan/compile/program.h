// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The program a primitive registers for the Vulkan runtime through
// RegisterDeviceProgram(vulkan, type, &def): its SPIR-V modules by name and
// file, and the pipelines it builds on a device context.
#pragma once
#include "usdGenVulkan/export.h"
#include "usdGen/api/span.h"
#include "usdGen/api/epoch.h"
#include "usdGen/api/pool.h"
#include "usdGen/api/ext/device/sessionDeviceProvider.h"

#include <functional>
#include <memory>
#include <string>

namespace usdGen {
struct UsdGenDiagnostics;
struct UsdGenGraphSpec;
class UsdGenExecutionRuntime;
class UsdGenExecutionPipeline;
}  // namespace usdGen

namespace usdGen::vulkan {

class DeviceContext;             // usdGenVulkan/vulkan/deviceContext.h
class ShaderLibrary;             // usdGenVulkan/exec/shaderLibrary.h
class VulkanCompletionService;   // usdGenVulkan/vulkan/completionService.h

/// One SPIR-V module a program needs: the name code uses, the file under the
/// shader directory.
struct ShaderDecl {
    char const *name;
    char const *file;
};

/// What the runtime owns for a program to run on: the device-owner pipeline,
/// the completion service and the identity the Session provider reports.
struct ProgramBinding {
    std::shared_ptr<UsdGenExecutionRuntime> runtime;
    std::shared_ptr<UsdGenExecutionPipeline> queueOwner;
    std::shared_ptr<VulkanCompletionService> completion;
    UsdGenSessionDeviceIdentity identity;
};

/// The provider-side half of a primitive's Vulkan program. The runtime's
/// Session provider shell forwards to it; the typed plan lives behind it.
/// Pipelines are built by ProgramDef::create; Bind adds the generation
/// domain and the submit route on the runtime's owner.
class USDGEN_VULKAN_API Program : public UsdGenRefCounted, public UsdGenPooled {
public:
    ~Program() override;
    virtual bool Bind(ProgramBinding, std::string *reason) = 0;
    virtual UsdGenSessionDevicePlanHandle Compile(UsdGenGraphSpec const &,
                                                  UsdGenDiagnostics *) const = 0;
    virtual UsdGenSessionDeviceIdentity Identity() const noexcept = 0;
    virtual bool NotifyContextLost(UsdGenSessionDeviceIdentity const &) noexcept = 0;
    /// False: not admitted.
    virtual bool Submit(UsdGenSessionDeviceRequest) = 0;
    /// Stops submit admission.
    virtual void Shutdown() noexcept = 0;
    /// Stops new generation leases (context loss, close).
    virtual void StopAdmission() noexcept = 0;
    /// External close boundary: drain the bounded CPU preparer.
    virtual void DrainPreparer() noexcept = 0;
    /// False while a lease is live; RetireWhenIdle then takes the close.
    virtual bool CloseDomain() noexcept = 0;
    virtual bool RetireWhenIdle(std::function<void()> externalRetirement) noexcept = 0;
};

struct ProgramDef {
    /// The first entry is the probe file that locates the shader directory.
    UsdGenSpan<const ShaderDecl> shaders;
    /// Any object in the registering module: the shader directory is
    /// resolved next to it.
    void const *moduleAnchor = nullptr;
    /// Builds the program's pipelines on a fresh context; null result with a
    /// reason when they cannot be built.
    UsdGenRef<Program> (*create)(std::shared_ptr<DeviceContext> const &,
                                 ShaderLibrary const &, std::string *reason) = nullptr;
};

/// The first program registered for the Vulkan backend that builds pipelines;
/// null when none is.
USDGEN_VULKAN_API ProgramDef const *FindCreatableProgram() noexcept;

}  // namespace usdGen::vulkan
