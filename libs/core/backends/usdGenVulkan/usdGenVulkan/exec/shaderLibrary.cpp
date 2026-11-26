// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenVulkan/exec/shaderLibrary.h"
#include "usdGenVulkan/backendId.h"
#include "usdGen/registries/backend.h"
#include "usdGen/registries/typeRegistry.h"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>

#if !defined(USDGEN_VULKAN_SHADER_RELATIVE_DIR)
#define USDGEN_VULKAN_SHADER_RELATIVE_DIR "vulkan"
#endif

namespace usdGen::vulkan {
namespace {

// Binary mode: text mode on Windows would rewrite 0x0D 0x0A words.
bool LoadSpirvFile(std::filesystem::path const &path, std::vector<uint32_t> *words)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.good()) return false;
    std::vector<char> raw((std::istreambuf_iterator<char>(file)), {});
    if (raw.empty() || raw.size() % sizeof(uint32_t) != 0) return false;
    words->resize(raw.size() / sizeof(uint32_t));
    std::memcpy(words->data(), raw.data(), raw.size());
    return true;
}

// Installed shaders live in share/usdGen/vulkan; build-tree shaders in
// <build>/vulkan. USDGEN_VULKAN_SHADER_DIR overrides both (tests).
std::filesystem::path ShaderDir(char const *probe, void const *anchor)
{
    if (char const *env = std::getenv("USDGEN_VULKAN_SHADER_DIR"))
        if (*env) return env;
    char module[4096];
    if (!anchor || !probe || !UsdGenModuleDirectory(anchor, module, sizeof(module))) return {};
    std::filesystem::path const parent(module);
    for (auto const &dir : {parent / "vulkan", parent / USDGEN_VULKAN_SHADER_RELATIVE_DIR}) {
        std::ifstream file(dir / probe, std::ios::binary);
        if (file.good()) return dir.lexically_normal();
    }
    return {};
}

}  // namespace

ShaderLibrary::ShaderLibrary() = default;
ShaderLibrary::~ShaderLibrary() = default;

void ShaderLibrary::Add(UsdGenSpan<const ShaderDecl> shaders, void const *moduleAnchor)
{
    if (shaders.empty()) return;
    std::filesystem::path const dir = ShaderDir(shaders[0].file, moduleAnchor);
    if (dir.empty()) return;
    for (ShaderDecl const &shader : shaders) {
        if (!shader.name || !shader.file || Has(shader.name)) continue;
        Entry entry;
        if (!LoadSpirvFile(dir / shader.file, &entry.words)) continue;
        entry.name = shader.name;
        _entries.push_back(std::move(entry));
    }
}

void ShaderLibrary::AddRegisteredPrograms()
{
    UsdGenBackendId const backend = UsdGenVulkanBackendId();
    UsdGenTypeRegistry const &types = UsdGenTypeRegistry::Get();
    for (uint32_t t = 0; t < types.Count(); ++t)
        if (auto const *def = static_cast<ProgramDef const *>(
                UsdGenFindDeviceProgram(backend, UsdGenTypeId(t))))
            Add(def->shaders, def->moduleAnchor);
}

bool ShaderLibrary::Has(char const *name) const noexcept
{
    return !Words(name).empty();
}

std::vector<uint32_t> const &ShaderLibrary::Words(char const *name) const noexcept
{
    if (name)
        for (Entry const &entry : _entries)
            if (entry.name == name) return entry.words;
    return _none;
}

}  // namespace usdGen::vulkan
