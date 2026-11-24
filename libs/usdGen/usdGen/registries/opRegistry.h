// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The op registry: immutable records, dense ids, frozen before the first
// compile. Registration goes through UsdGenRegistrar (registrar.h).
#pragma once
#include "usdGen/export.h"
#include "usdGen/registries/opDef.h"
#include "usdGen/curveBuffer.h"          // UsdGenPlane (step 5 replaces it)
#include "pxr/base/tf/token.h"
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE
namespace usdGen {

class UsdGenRegistrar;
USDGEN_CORE_API void UsdGenFreezeRegistries();   // registrar.h

/// Immutable after RegisterOp. Every name interned once.
class USDGEN_CORE_API UsdGenOpRecord {
public:
    UsdGenOpId Id() const noexcept { return _id; }
    TfToken const &Name() const noexcept { return _name; }
    UsdGenOpDef const &Def() const noexcept { return _def; }
    bool Has(uint32_t trait) const noexcept { return (_def.traits & trait) != 0; }

    /// Accepts "x" and "usdGen:x". Pointer compares; never interns.
    UsdGenParamSlot FindParam(TfToken const &name) const noexcept;
    uint16_t ParamCount() const noexcept { return uint16_t(_def.params.size()); }
    TfToken const &ParamName(UsdGenParamSlot) const noexcept;
    UsdGenParamClass ParamClass(UsdGenParamSlot s) const noexcept { return _def.params[s].cls; }
    int EnumIndex(UsdGenParamSlot, TfToken const &value) const noexcept;     // -1: no match
    int EnumIndex(UsdGenParamSlot, std::string const &value) const noexcept;
    UsdGenParamSlot EnabledSlot() const noexcept { return _enabled; }        // kUsdGenNoParam if undeclared

    TfSpan<const std::pair<TfToken, uint32_t>> Routing() const noexcept;    // == TypeClassTable::routing
    TfSpan<const UsdGenParamSlot> DigestSlots() const noexcept;             // == TypeClassTable::digestParams
    TfSpan<const TfToken> ReferenceInputs() const noexcept;
    TfSpan<const UsdGenPlane> InputPlanes() const noexcept;                 // layouts filled, no payload
    TfSpan<const UsdGenPlane> OutputPlanes() const noexcept;                // static outputs only
    TfSpan<const TfToken> MapRelationships() const noexcept;
    TfToken const &Token(uint16_t i) const noexcept;                        // Def().tokens[i]

private:
    friend class UsdGenRegistrar;
    UsdGenOpId _id = UsdGenOpId::Invalid;
    TfToken _name;
    UsdGenOpDef _def;
    UsdGenParamSlot _enabled = kUsdGenNoParam;
    struct Tables;                       // interned names, hash slots, routing; built at registration
    Tables const *_tables = nullptr;
};

class USDGEN_CORE_API UsdGenOpRegistry {
public:
    /// CAS-published, as before. Non-const until the tests stop calling the
    /// legacy Register below (S4-T); then const.
    static UsdGenOpRegistry &Get() noexcept;
    bool IsFrozen() const noexcept { return _frozen.load(std::memory_order_acquire); }
    uint32_t Count() const noexcept { return _count; }
    /// Compile phase only. Pointer-hash probe; never interns.
    UsdGenOpId Find(TfToken const &name) const noexcept;
    UsdGenOpRecord const &Record(UsdGenOpId id) const noexcept { return _records[UsdGenIndex(id)]; }
    std::unique_ptr<UsdGenOp> Create(UsdGenOpId id) const;

    // ---- Legacy API (temporary: S4-T moves the tests, S4-8 deletes it) ----
    using Factory = std::function<std::unique_ptr<UsdGenOp>()>;
    /// RegisterLegacyOp. False on a duplicate type or after freeze.
    bool Register(TfToken const &type, Factory factory);
    std::unique_ptr<UsdGenOp> Create(TfToken const &type) const;
    bool HasKernel(TfToken const &type) const;
    bool ReadsTime(TfToken const &type) const;
    bool SamplesFrameInputs(TfToken const &type) const;
    bool GetOperatorContract(TfToken const &type, size_t *outGeometryInputArity,
                             size_t *outReferenceInputArity, UsdGenRole *outRole) const;
    bool GetGeometryInputArity(TfToken const &type, size_t *outArity) const;
    /// Registered type names in registration order.
    std::vector<TfToken> KnownTypes() const;

private:
    friend class UsdGenRegistrar;
    friend void UsdGenFreezeRegistries();
    UsdGenOpRegistry();
    UsdGenOpRecord _records[kUsdGenMaxOps];
    uint32_t _count = 0;
    struct NameSlot { TfToken name; UsdGenOpId id = UsdGenOpId::Invalid; };
    NameSlot _byName[2 * kUsdGenMaxOps];
    std::atomic<bool> _frozen{false};
};

/// Fills UsdGenNodeDesc::opId and UsdGenMapDesc::typeId. Compile phase.
USDGEN_CORE_API void UsdGenResolveDescIds(UsdGenGraphDesc *desc);

/// Generic expression admission over the record's decls; replaces
/// expressionTargets.cpp. Messages (substrings tests check today):
///   "unsupported or incorrectly typed <displayName> expression <param>"
///   "<displayName> <param> requires groom evaluation"
///   "<displayName> <param> requires groom/primitive evaluation"
///   "operator <name> at <path> does not accept connected (expression) parameters"
USDGEN_CORE_API bool UsdGenValidateExpressionBindings(UsdGenOpRecord const &,
    UsdGenNodeDesc const &, std::vector<std::string> *errors);

/// Legacy (S4-8 deletes): makes sure the built-ins are registered.
USDGEN_CORE_API void usdGenRegisterM1Operators();

}  // namespace usdGen
