// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Key function of the program interface: one typeinfo, in usdGenVulkan.
#include "usdGenVulkan/compile/program.h"
#include "usdGenVulkan/backendId.h"
#include "usdGen/api/ext/backendDef.h"
#include "usdGen/api/ext/typeRegistry.h"

namespace usdGen::vulkan {

Program::~Program() = default;

ProgramDef const *FindCreatableProgram() noexcept
{
    UsdGenBackendId const backend = UsdGenVulkanBackendId();
    UsdGenTypeRegistry const &types = UsdGenTypeRegistry::Get();
    for (uint32_t t = 0; t < types.Count(); ++t)
        if (auto const *def = static_cast<ProgramDef const *>(
                UsdGenFindDeviceProgram(backend, UsdGenTypeId(t))))
            if (def->create) return def;
    return nullptr;
}

}  // namespace usdGen::vulkan
