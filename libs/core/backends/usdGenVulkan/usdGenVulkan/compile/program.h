// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The program a primitive registers for the Vulkan runtime through
// RegisterDeviceProgram(vulkan, type, &def): its SPIR-V modules by name and
// file, and the pipelines it builds on a device context.
#pragma once
#include "usdGenVulkan/export.h"
#include "usdGen/api/span.h"
#include "usdGen/exec/epoch.h"
#include "usdGen/exec/pool.h"
#include "usdGen/exec/sessionDeviceProvider.h"

#include <memory>
#include <string>

namespace usdGen {
struct UsdGenDiagnostics;
struct UsdGenGraphSpec;
}  // namespace usdGen

namespace usdGen::vulkan {

class DeviceContext;   // usdGenVulkan/vulkan/deviceContext.h
class ShaderLibrary;   // usdGenVulkan/exec/shaderLibrary.h

/// One SPIR-V module a program needs: the name code uses, the file under the
/// shader directory.
struct ShaderDecl {
    char const *name;
    char const *file;
};

class USDGEN_VULKAN_API Program : public UsdGenRefCounted, public UsdGenPooled {
public:
    ~Program() override;
    virtual UsdGenSessionDevicePlanHandle Compile(UsdGenGraphSpec const &,
                                                  UsdGenDiagnostics *) const = 0;
    /// False: not admitted.
    virtual bool Submit(UsdGenSessionDeviceRequest) = 0;
    virtual void Shutdown() noexcept = 0;
};

struct ProgramDef {
    /// The first entry is the probe file that locates the shader directory.
    UsdGenSpan<const ShaderDecl> shaders;
    /// Any object in the registering module: the shader directory is
    /// resolved next to it.
    void const *moduleAnchor = nullptr;
    /// Builds the program's pipelines on a fresh context. Null: the runtime
    /// still builds them (until the pipelines move to their primitive).
    UsdGenRef<Program> (*create)(std::shared_ptr<DeviceContext> const &,
                                 ShaderLibrary const &, std::string *reason) = nullptr;
};

}  // namespace usdGen::vulkan
