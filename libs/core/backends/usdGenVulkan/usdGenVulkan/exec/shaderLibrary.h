// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// SPIR-V of the registered Vulkan programs, loaded once per device context
// and served by name.
#pragma once
#include "usdGenVulkan/export.h"
#include "usdGenVulkan/compile/program.h"
#include "usdGen/api/span.h"

#include <cstdint>
#include <string>
#include <vector>

namespace usdGen::vulkan {

class USDGEN_VULKAN_API ShaderLibrary {
public:
    ShaderLibrary();
    ~ShaderLibrary();
    ShaderLibrary(ShaderLibrary const &) = delete;
    ShaderLibrary &operator=(ShaderLibrary const &) = delete;

    /// Loads the declared files from USDGEN_VULKAN_SHADER_DIR, else from
    /// <anchor's module>/vulkan or <anchor's module>/<installed relative
    /// dir>, whichever holds the first declared file. A missing or malformed
    /// file leaves its name absent; a name already loaded is kept.
    void Add(UsdGenSpan<const ShaderDecl> shaders, void const *moduleAnchor);
    /// Add for every program registered for the Vulkan backend.
    void AddRegisteredPrograms();

    bool Has(char const *name) const noexcept;
    /// Empty when absent.
    std::vector<uint32_t> const &Words(char const *name) const noexcept;

private:
    struct Entry {
        std::string name;
        std::vector<uint32_t> words;
    };
    std::vector<Entry> _entries;
    std::vector<uint32_t> _none;
};

}  // namespace usdGen::vulkan
