// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/api/ext/registrar.h"
#include "usdGen/api/elementSet.h"
#include "usdGen/api/engineTokens.h"
#include "usdGen/registries/opRecordTables.h"
#include "usdGen/api/ext/tokenScope.h"

#include <cstdlib>
#include <cstring>
#include <fstream>
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

/// Element level of a plane rate: vertex 2, uniform 1, constant 0.
uint8_t PlaneLevel(UsdGenPlaneRate rate)
{
    return rate == UsdGenPlaneRate::Vertex ? 2 : rate == UsdGenPlaneRate::Uniform ? 1 : 0;
}

void FillPlanes(UsdGenSpan<const UsdGenPlaneDecl> decls, std::vector<UsdGenChannelDesc> *out)
{
    for (UsdGenPlaneDecl const &d : decls) {
        UsdGenChannelDesc desc;
        desc.name = UsdGenInternToken(d.name);
        desc.level = PlaneLevel(d.rate);
        desc.scalar = d.scalar == UsdGenPlaneScalar::Int ? UsdGenScalar::I32 : UsdGenScalar::F32;
        desc.arity = d.arity;
        desc.flags = d.arity > 1 ? uint8_t(kUsdGenChannelPacked) : uint8_t(0);
        out->push_back(desc);
    }
}

bool HashInsert(std::vector<std::pair<UsdGenToken, UsdGenParamSlot>> *hash, size_t mask,
                UsdGenToken const &key, UsdGenParamSlot slot)
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
uint32_t RoutingBits(UsdGenEngineTokens const &t, UsdGenToken const &name,
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
    UsdGenToken const name = UsdGenInternToken(def.name);
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
        UsdGenToken const param = UsdGenInternToken(decl.name);
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
    // "usdGen:x" finds x unless a decl spells "usdGen:x" itself (an op
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
        tables->inputTypes.push_back(type && *type ? UsdGenInternToken(type) : UsdGenToken());
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
    UsdGenToken const name = UsdGenInternToken(def.name);
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
    char buffer[4096];
    size_t const n = UsdGenModuleDirectory(&kModuleAnchor, buffer, sizeof(buffer));
    return std::string(buffer, n);
}

// Set while this thread runs the default extensions' usdGenRegister.
thread_local bool tRegistering = false;

bool IsAbsolute(std::string const &path)
{
    return (!path.empty() && (path[0] == '/' || path[0] == '\\')) ||
           (path.size() > 1 && path[1] == ':');
}

// The directory part of 'path' with its trailing separator; empty if none.
std::string DirectoryOf(std::string const &path)
{
    size_t const slash = path.find_last_of("/\\");
    return slash == std::string::npos ? std::string() : path.substr(0, slash + 1);
}

// One file name per line, in load order, relative to the manifest's
// directory. Blank lines and '#' lines are skipped. False if unreadable.
bool ReadManifest(std::string const &manifest, std::vector<std::string> *paths)
{
    std::ifstream in(manifest, std::ios::binary);
    if (!in) return false;
    std::string const dir = DirectoryOf(manifest);
    std::string line;
    while (std::getline(in, line)) {
        size_t begin = 0, end = line.size();
        while (begin < end && (line[begin] == ' ' || line[begin] == '\t')) ++begin;
        while (end > begin && (line[end - 1] == ' ' || line[end - 1] == '\t' || line[end - 1] == '\r'))
            --end;
        if (begin == end || line[begin] == '#') continue;
        std::string name = line.substr(begin, end - begin);
        paths->push_back(IsAbsolute(name) ? std::move(name) : dir + name);
    }
    return true;
}

}  // namespace

bool UsdGenRegistering() noexcept { return tRegistering; }

size_t UsdGenModuleDirectory(void const *anchor, char *buffer, size_t capacity) noexcept
{
    if (!buffer || capacity == 0) return 0;
    buffer[0] = 0;
    size_t n = 0;
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            static_cast<LPCSTR>(anchor), &module))
        return 0;
    DWORD const got = GetModuleFileNameA(module, buffer, DWORD(capacity));
    if (got == 0 || size_t(got) >= capacity) {
        buffer[0] = 0;
        return 0;
    }
    n = size_t(got);
#else
    Dl_info info;
    if (!dladdr(anchor, &info) || !info.dli_fname) return 0;
    n = std::strlen(info.dli_fname);
    if (n >= capacity) return 0;
    std::memcpy(buffer, info.dli_fname, n);
#endif
    while (n > 0 && buffer[n - 1] != '/' && buffer[n - 1] != '\\') --n;
    buffer[n] = 0;
    return n;
}

// The op registry's constructor: registers the extensions 'manifest' names,
// else USDGEN_EXTENSIONS, else usdGenExtensions.txt beside usdGen, in order.
// Keeps going past a failure; 'error' gets the first.
void UsdGenLoadDefaultExtensions(UsdGenRegistrar &registrar, char const *manifest,
                                 std::string *error)
{
#if defined(_WIN32)
    char const separator = ';';
#else
    char const separator = ':';
#endif
    std::vector<std::string> paths;
    if (manifest) {
        if (!ReadManifest(manifest, &paths))
            *error = std::string("cannot read the extension manifest '") + manifest + "'";
    } else if (char const *list = std::getenv("USDGEN_EXTENSIONS")) {
        std::string const all(list);
        for (size_t begin = 0; begin <= all.size();) {
            size_t end = all.find(separator, begin);
            if (end == std::string::npos) end = all.size();
            if (end > begin) paths.push_back(all.substr(begin, end - begin));
            begin = end + 1;
        }
    } else {
        // Written by CMake beside usdGen (and installed with it); absent: no extension.
        std::string const dir = EngineDirectory();
        if (dir.empty()) *error = "cannot locate usdGen's directory";
        else (void)ReadManifest(dir + "usdGenExtensions.txt", &paths);
    }
    for (std::string const &path : paths) {
        std::string message;
        UsdGenRegisterFn const fn = OpenExtension(path, IsAbsolute(path), &message);
        if (fn) {
            tRegistering = true;
            fn(registrar);
            tRegistering = false;
        }
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
