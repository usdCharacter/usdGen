// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Input types the engine branches on (step 4: the map prim types).
#pragma once
#include "usdGen/export.h"
#include "usdGen/api/ids.h"
#include "usdGen/registries/handles.h"
#include "pxr/base/tf/token.h"
#include <atomic>
#include <cstdint>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE
namespace usdGen {

struct UsdGenMapInputView;
struct UsdGenElementLayout;
struct UsdGenInputDesc;
class UsdGenTypeRegistry;
USDGEN_CORE_API void UsdGenFreezeRegistries();   // registrar.h
UsdGenTypeRegistry &UsdGenMutableTypeRegistry() noexcept;   // registrar-only

enum UsdGenTypeTrait : uint32_t {
    kUsdGenTypeMap               = 1u << 0,
    kUsdGenTypeMapPaintSnapshot  = 1u << 1,   // graph.cpp:199, sessionCooker.cpp:467, cpuParameters.cpp:422, strandUtil.cpp:407
    kUsdGenTypeMapPtex           = 1u << 2,   // surfaceCageSource.cpp:154, valuePreview.cpp:368, regionMap.h:31
    kUsdGenTypeMapImage          = 1u << 3,   // strandUtil.cpp:415
    kUsdGenTypeMapExprSampleable = 1u << 4,   // cpuParameters.cpp:418 (ptex() accepts it)
};

/// Map prim parameters parsed once (cpuParameters.cpp:424-436,
/// surfaceCageSource.cpp:159-190, ops/regionMap.h:35-42).
struct UsdGenMapOptions {
    std::string filter, borderMode, channel = "r";
    float blur = 0.0f;
    int firstChannel = 0, channelCount = 1;
    double scale = 1.0, offset = 0.0, fallback = 0.0, clampLo = 0.0, clampHi = 1.0;
    bool clamp = false;
    bool filterAuthored = false, borderModeAuthored = false, blurAuthored = false;
    uint64_t textureGeneration = 0;      // "map:textureGeneration" (surfaceCageSource.cpp:159)
};

struct UsdGenTypeDef {
    char const *name = nullptr;          // e.g. UsdGenPtexMap
    uint32_t traits = 0;
    bool (*parseOptions)(UsdGenMapInputView const &, UsdGenMapOptions *, std::string *error) = nullptr;
    /// Primitive types (UsdGenCurveSet, UsdGenRootSet): the element layout;
    /// null for map and value types. Must outlive the registry.
    UsdGenElementLayout const *layout = nullptr;
    uint32_t limits[kUsdGenMaxLevels] = {};   // max items per parent; curves: {0, 4096, 0}
    /// Input kinds (UsdGenCurveSetInput, ...): how a host fills a record of
    /// this type ('layout' is the record's). Must outlive the registry.
    UsdGenInputDesc const *input = nullptr;
};

class USDGEN_CORE_API UsdGenTypeRegistry {
public:
    static UsdGenTypeRegistry const &Get() noexcept;
    UsdGenTypeId Find(TfToken const &name) const noexcept;   // compile phase only
    UsdGenTypeDef const &Def(UsdGenTypeId) const noexcept;
    TfToken const &Name(UsdGenTypeId) const noexcept;
    uint32_t Count() const noexcept { return _count; }
    bool Has(UsdGenTypeId id, uint32_t trait) const noexcept {
        return id != UsdGenTypeId::Invalid && (Def(id).traits & trait) != 0;
    }
private:
    friend class UsdGenRegistrar;
    friend void UsdGenFreezeRegistries();
    friend UsdGenTypeRegistry &UsdGenMutableTypeRegistry() noexcept;
    UsdGenTypeRegistry();
    UsdGenTypeDef _defs[kUsdGenMaxTypes];
    TfToken _names[kUsdGenMaxTypes];
    uint32_t _count = 0;
    std::atomic<bool> _frozen{false};
};

}  // namespace usdGen
