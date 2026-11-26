// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/registries/registrar.h"
#include "usdGen/api/elementSet.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/opRecordTables.h"
#include "usdGen/registries/token.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace usdGen {

UsdGenTypeRegistry &UsdGenMutableTypeRegistry() noexcept;          // typeRegistry.cpp
UsdGenBackendRegistry &UsdGenMutableBackendRegistry() noexcept;    // backendRegistry.cpp
void UsdGenFreezeExecutionBackendRegistry() noexcept;              // executionBackend.h

namespace {

// Written by registration only (before the freeze); read after it.
UsdGenPublisherDef gPublishers[kUsdGenMaxTypes];

TfToken InternPlaneRate(UsdGenPlaneRate rate)
{
    UsdGenEngineTokens const &t = UsdGenTokens();
    return rate == UsdGenPlaneRate::Vertex ? t.vertex
        : rate == UsdGenPlaneRate::Uniform ? t.uniform : t.constant;
}

void FillPlanes(TfSpan<const UsdGenPlaneDecl> decls, std::vector<UsdGenPlane> *out)
{
    UsdGenEngineTokens const &t = UsdGenTokens();
    for (UsdGenPlaneDecl const &d : decls) {
        UsdGenPlane plane;
        plane.name = UsdGenInternToken(d.name);
        plane.interpolation = InternPlaneRate(d.rate);
        plane.type = d.scalar == UsdGenPlaneScalar::Int ? t.int_ : t.float_;
        plane.arity = d.arity;
        out->push_back(std::move(plane));
    }
}

bool HashInsert(std::vector<std::pair<TfToken, UsdGenParamSlot>> *hash, size_t mask,
                TfToken const &key, UsdGenParamSlot slot)
{
    for (size_t i = key.Hash() & mask;; i = (i + 1) & mask) {
        auto &entry = (*hash)[i];
        if (entry.first.IsEmpty()) {
            entry = {key, slot};
            return true;
        }
        if (entry.first == key) return false;
    }
}

/// Dirty bits of one routed parameter: compiler.cpp ClassifyParamBits.
uint32_t RoutingBits(UsdGenEngineTokens const &t, TfToken const &name,
                     UsdGenParamClass cls, uint8_t countChanges)
{
    if (cls == UsdGenParamClass::Value) return UsdGenDirtyParameter;
    if (name == t.input || name == t.guides || name == t.surface)
        return UsdGenDirtyStructural;
    if (name == t.enabled)
        return countChanges != 0 ? (UsdGenDirtyTopology | UsdGenDirtyCapture)
                                            : UsdGenDirtyParameter;
    if (name == t.seed) return UsdGenDirtyCapture;
    switch (cls) {
    case UsdGenParamClass::Structural: return UsdGenDirtyStructural;
    case UsdGenParamClass::Topology: return UsdGenDirtyTopology;
    default: return UsdGenDirtyCapture;
    }
}

}  // namespace

UsdGenOpId UsdGenRegistrar::RegisterOp(UsdGenOpDef const &def)
{
    UsdGenOpRegistry &ops = *_ops;
    if (ops.IsFrozen() || !def.name || !*def.name || !def.create ||
        ops._count >= kUsdGenMaxOps || def.params.size() > kUsdGenMaxParamsPerOp ||
        def.tokens.size() > kUsdGenMaxOpTokens)
        return UsdGenOpId::Invalid;
    UsdGenCompileTokenScope scope;
    TfToken const name = UsdGenInternToken(def.name);
    if (ops.Find(name) != UsdGenOpId::Invalid) return UsdGenOpId::Invalid;

    UsdGenEngineTokens const &et = UsdGenTokens();
    auto tables = std::make_unique<UsdGenOpRecord::Tables>();
    size_t const n = def.params.size();
    size_t cap = 8;
    while (cap < 4 * n) cap *= 2;
    tables->hash.resize(cap);
    tables->mask = cap - 1;
    tables->paramNames.reserve(n);
    tables->enumerants.resize(n);
    UsdGenParamSlot enabled = kUsdGenNoParam;
    for (size_t s = 0; s < n; ++s) {
        UsdGenParamDecl const &decl = def.params[s];
        if (!decl.name || !*decl.name) return UsdGenOpId::Invalid;
        TfToken const param = UsdGenInternToken(decl.name);
        if (!HashInsert(&tables->hash, tables->mask, param, UsdGenParamSlot(s)))
            return UsdGenOpId::Invalid;
        for (size_t e = 0; e < decl.enumerants.size(); ++e) {
            char const *value = decl.enumerants[e];
            if (!value) return UsdGenOpId::Invalid;
            for (size_t k = 0; k < e; ++k)
                if (std::strcmp(decl.enumerants[k], value) == 0) return UsdGenOpId::Invalid;
            tables->enumerants[s].push_back(UsdGenInternToken(value));
        }
        if (param == et.enabled) enabled = UsdGenParamSlot(s);
        if (decl.routed)
            tables->routing.push_back({param, RoutingBits(et, param, decl.cls, def.countChanges)});
        if (decl.routed && decl.cls == UsdGenParamClass::Structural)
            tables->digest.push_back(UsdGenParamSlot(s));
        tables->paramNames.push_back(param);
    }
    // "usdGen:x" finds x unless a decl spells "usdGen:x" itself (CurveSource
    // declares both useRest and usdGen:useRest).
    for (size_t s = 0; s < n; ++s) {
        char const *const declName = def.params[s].name;
        if (std::strncmp(declName, "usdGen:", 7) == 0) continue;
        HashInsert(&tables->hash, tables->mask,
                   UsdGenInternToken(std::string("usdGen:") + declName), UsdGenParamSlot(s));
    }
    for (char const *ref : def.referenceInputs) {
        if (!ref) return UsdGenOpId::Invalid;
        tables->referenceInputs.push_back(UsdGenInternToken(ref));
    }
    for (UsdGenMapSlotDecl const &slot : def.mapSlots) {
        if (!slot.relationship) return UsdGenOpId::Invalid;
        tables->mapRelationships.push_back(UsdGenInternToken(slot.relationship));
    }
    for (char const *token : def.tokens) {
        if (!token) return UsdGenOpId::Invalid;
        tables->tokens.push_back(UsdGenInternToken(token));
    }
    if (def.outputType && *def.outputType) tables->outputType = UsdGenInternToken(def.outputType);
    for (char const *type : def.inputTypes)
        tables->inputTypes.push_back(type && *type ? UsdGenInternToken(type) : TfToken());
    FillPlanes(def.inputPlanes, &tables->inputPlanes);
    FillPlanes(def.outputPlanes, &tables->outputPlanes);

    UsdGenOpId const id = UsdGenOpId(ops._count);
    UsdGenOpRecord &record = ops._records[ops._count];
    record._id = id;
    record._name = name;
    record._def = def;
    record._enabled = enabled;
    record._tables = tables.release();   // lives as long as the registry
    constexpr size_t mask = 2 * kUsdGenMaxOps - 1;
    for (size_t i = name.Hash() & mask;; i = (i + 1) & mask) {
        if (ops._byName[i].name.IsEmpty()) {
            ops._byName[i].name = name;
            ops._byName[i].id = id;
            break;
        }
    }
    ++ops._count;
    return id;
}

UsdGenTypeId UsdGenRegistrar::RegisterType(UsdGenTypeDef const &def)
{
    UsdGenTypeRegistry &types = UsdGenMutableTypeRegistry();
    if (_ops->IsFrozen() || types._frozen.load(std::memory_order_acquire) || !def.name ||
        !*def.name || types._count >= kUsdGenMaxTypes)
        return UsdGenTypeId::Invalid;
    if (def.layout && (def.layout->levels < 1 || def.layout->levels > kUsdGenMaxLevels ||
                       def.layout->channels.size() >= kUsdGenNoSlot))
        return UsdGenTypeId::Invalid;
    UsdGenCompileTokenScope scope;
    TfToken const name = UsdGenInternToken(def.name);
    if (types.Find(name) != UsdGenTypeId::Invalid) return UsdGenTypeId::Invalid;
    types._defs[types._count] = def;
    types._names[types._count] = name;
    return UsdGenTypeId(types._count++);
}

bool UsdGenRegistrar::RegisterKernel(UsdGenOpId op, UsdGenBackendId backend,
                                     UsdGenKernelDef const &kernel)
{
    UsdGenBackendRegistry &kernels = UsdGenMutableBackendRegistry();
    uint32_t const b = UsdGenIndex(backend), o = UsdGenIndex(op);
    if (_ops->IsFrozen() || kernels._frozen.load(std::memory_order_acquire) ||
        b >= kUsdGenMaxBackends || o >= _ops->_count || kernels._present[b][o])
        return false;
    kernels._kernels[b][o] = kernel;
    kernels._present[b][o] = true;
    kernels._order[b][kernels._orderCount[b]++] = op;
    return true;
}

bool UsdGenRegistrar::RegisterParamEvaluator(UsdGenParamEvaluatorDef const &def)
{
    UsdGenParamEvaluatorDef &slot = _ops->_paramEvaluator;
    if (_ops->IsFrozen() || !def.create || slot.create) return false;
    slot = def;
    return true;
}

bool UsdGenRegistrar::RegisterPublisher(UsdGenPublisherDef const &def)
{
    uint32_t const t = UsdGenIndex(def.primitive);
    if (_ops->IsFrozen() || !def.build || def.primitive == UsdGenTypeId::Invalid ||
        t >= UsdGenMutableTypeRegistry().Count() || gPublishers[t].build)
        return false;
    gPublishers[t] = def;
    return true;
}

UsdGenPublisherDef const *UsdGenFindPublisher(UsdGenTypeId primitive) noexcept
{
    uint32_t const t = UsdGenIndex(primitive);
    return t < kUsdGenMaxTypes && gPublishers[t].build ? &gPublishers[t] : nullptr;
}

std::unique_ptr<UsdGenParamEvaluatorState> UsdGenCreateParamEvaluatorState()
{
    UsdGenParamEvaluatorDef const &def = UsdGenOpRegistry::Get().ParamEvaluator();
    return def.create ? def.create() : nullptr;
}

bool UsdGenEngineEnsureInitialized(std::string *error)
{
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    if (error) *error = ops.InitError();
    return ops.InitError().empty();
}

bool UsdGenRegisterStatic(UsdGenRegisterFn fn)
{
    UsdGenOpRegistry &ops = UsdGenMutableOpRegistry();
    if (!fn || ops.IsFrozen()) return false;
    UsdGenRegistrar registrar(ops);
    fn(registrar);
    return true;
}

namespace {

// Opens 'path' and returns its usdGenRegister. The module stays loaded:
// registered defs point into it. 'siblings' resolves the module's own
// dependencies from its directory (absolute paths only).
UsdGenRegisterFn OpenExtension(std::string const &path, bool siblings, std::string *error)
{
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (siblings) {
        std::string native = path;   // the altered search wants backslashes
        for (char &c : native)
            if (c == '/') c = '\\';
        module = LoadLibraryExA(native.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    } else {
        module = LoadLibraryA(path.c_str());
    }
    if (!module) {
        *error = "cannot load '" + path + "'";
        return nullptr;
    }
    auto fn = reinterpret_cast<UsdGenRegisterFn>(GetProcAddress(module, "usdGenRegister"));
#else
    (void)siblings;
    void *module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!module) {
        *error = std::string("cannot load '") + path + "': " + dlerror();
        return nullptr;
    }
    auto fn = reinterpret_cast<UsdGenRegisterFn>(dlsym(module, "usdGenRegister"));
#endif
    if (!fn) *error = "'" + path + "' exports no usdGenRegister";
    return fn;
}

// Any object of this module; its address names usdGen's own file.
char const kModuleAnchor = 0;

// usdGen's directory with a trailing separator; empty if unknown.
std::string EngineDirectory()
{
#if defined(_WIN32)
    HMODULE self = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            &kModuleAnchor, &self))
        return std::string();
    char buffer[4096];
    DWORD const n = GetModuleFileNameA(self, buffer, DWORD(sizeof(buffer)));
    if (n == 0 || n >= sizeof(buffer)) return std::string();
    std::string file(buffer, n);
#else
    Dl_info info;
    if (!dladdr(&kModuleAnchor, &info) || !info.dli_fname) return std::string();
    std::string file(info.dli_fname);
#endif
    size_t const slash = file.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : file.substr(0, slash + 1);
}

}  // namespace

// The op registry's constructor: registers the default extensions (or
// USDGEN_EXTENSIONS) in order. Keeps going past a failure; 'error' gets the first.
void UsdGenLoadDefaultExtensions(UsdGenRegistrar &registrar, std::string *error)
{
#if defined(_WIN32)
    char const separator = ';';
#else
    char const separator = ':';
#endif
    std::vector<std::string> paths;
    if (char const *list = std::getenv("USDGEN_EXTENSIONS")) {
        std::string const all(list);
        for (size_t begin = 0; begin <= all.size();) {
            size_t end = all.find(separator, begin);
            if (end == std::string::npos) end = all.size();
            if (end > begin) paths.push_back(all.substr(begin, end - begin));
            begin = end + 1;
        }
    } else {
        std::string const dir = EngineDirectory();
        char const *const files[] = {
#if defined(USDGEN_SURFACE_FILE) && defined(USDGEN_EXPR_FILE) && defined(USDGEN_CURVES_FILE)
            USDGEN_SURFACE_FILE, USDGEN_EXPR_FILE, USDGEN_CURVES_FILE,
#endif
            nullptr};
        if (dir.empty() && files[0]) *error = "cannot locate usdGen's directory";
        for (char const *const *f = files; *f && !dir.empty(); ++f) paths.push_back(dir + *f);
    }
    for (std::string const &path : paths) {
        std::string message;
        bool const absolute = (!path.empty() && (path[0] == '/' || path[0] == '\\')) ||
                              (path.size() > 1 && path[1] == ':');
        UsdGenRegisterFn const fn = OpenExtension(path, absolute, &message);
        if (fn) fn(registrar);
        else if (error->empty()) *error = std::move(message);
    }
}

bool UsdGenLoadExtension(std::string const &path, std::string *error)
{
    auto fail = [error](std::string message) {
        if (error) *error = std::move(message);
        return false;
    };
    if (UsdGenOpRegistry::Get().IsFrozen())
        return fail("registries are frozen; load extensions before the first compile");
    std::string message;
    UsdGenRegisterFn const fn = OpenExtension(path, false, &message);
    if (!fn) return fail(std::move(message));
    return UsdGenRegisterStatic(fn);
}

void UsdGenFreezeRegistries()
{
    UsdGenOpRegistry &ops = UsdGenMutableOpRegistry();
    if (ops.IsFrozen()) return;
    UsdGenMutableTypeRegistry()._frozen.store(true, std::memory_order_release);
    UsdGenMutableBackendRegistry()._frozen.store(true, std::memory_order_release);
    UsdGenFreezeExecutionBackendRegistry();
    ops._frozen.store(true, std::memory_order_release);
}

}  // namespace usdGen
