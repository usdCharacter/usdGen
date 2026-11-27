// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The op registry: immutable records, dense ids, frozen before the first
// compile. Registration goes through UsdGenRegistrar (registrar.h).
#pragma once
#include "usdGen/api/export.h"
#include "usdGen/registries/opDef.h"
#include "usdGen/registries/paramEvaluator.h"
#include "usdGen/api/channel.h"
#include "usdGen/api/token.h"
#include <atomic>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace usdGen {

class UsdGenRegistrar;
class UsdGenOpRegistry;
USDGEN_CORE_API void UsdGenFreezeRegistries();   // registrar.h
UsdGenOpRegistry &UsdGenMutableOpRegistry() noexcept;   // registrar-only
UsdGenOpRegistry &UsdGenBuildOpRegistry(char const *manifest);   // UsdGenEngine::Create only

/// Immutable after RegisterOp. Every name interned once.
class USDGEN_CORE_API UsdGenOpRecord {
public:
    UsdGenOpId Id() const noexcept { return _id; }
    UsdGenToken const &Name() const noexcept { return _name; }
    UsdGenOpDef const &Def() const noexcept { return _def; }
    bool Has(uint32_t trait) const noexcept { return (_def.traits & trait) != 0; }

    /// Accepts "x" and "usdGen:x". Pointer compares; never interns.
    UsdGenParamSlot FindParam(UsdGenToken const &name) const noexcept;
    uint16_t ParamCount() const noexcept { return uint16_t(_def.params.size()); }
    UsdGenToken const &ParamName(UsdGenParamSlot) const noexcept;
    UsdGenParamClass ParamClass(UsdGenParamSlot s) const noexcept { return _def.params[s].cls; }
    int EnumIndex(UsdGenParamSlot, UsdGenToken const &value) const noexcept;     // -1: no match
    int EnumIndex(UsdGenParamSlot, std::string const &value) const noexcept;
    UsdGenParamSlot EnabledSlot() const noexcept { return _enabled; }        // kUsdGenNoParam if undeclared

    UsdGenSpan<const std::pair<UsdGenToken, uint32_t>> Routing() const noexcept;    // == TypeClassTable::routing
    UsdGenSpan<const UsdGenParamSlot> DigestSlots() const noexcept;             // == TypeClassTable::digestParams
    UsdGenSpan<const UsdGenToken> ParamNames() const noexcept;                          // per slot, no "usdGen:"
    UsdGenSpan<const UsdGenToken> ReferenceInputs() const noexcept;
    UsdGenSpan<const UsdGenChannelDesc> InputPlanes() const noexcept;           // declared layouts
    UsdGenSpan<const UsdGenChannelDesc> OutputPlanes() const noexcept;          // static outputs only
    UsdGenSpan<const UsdGenToken> MapRelationships() const noexcept;
    UsdGenToken const &Token(uint16_t i) const noexcept;                        // Def().tokens[i]
    /// Typed ports (interned names; empty: the default primitive).
    UsdGenToken const &OutputType() const noexcept;
    UsdGenSpan<const UsdGenToken> InputTypes() const noexcept;                      // empty: the default only
    bool AcceptsInput(UsdGenToken const &producerType) const noexcept;

private:
    friend class UsdGenRegistrar;
    UsdGenOpId _id = UsdGenOpId::Invalid;
    UsdGenToken _name;
    UsdGenOpDef _def;
    UsdGenParamSlot _enabled = kUsdGenNoParam;
    struct Tables;                       // interned names, hash slots, routing; built at registration
    Tables const *_tables = nullptr;
};

class USDGEN_CORE_API UsdGenOpRegistry {
public:
    /// Built by UsdGenEngine::Create (called on first use) and published by CAS.
    static UsdGenOpRegistry const &Get() noexcept;
    bool IsFrozen() const noexcept { return _frozen.load(std::memory_order_acquire); }
    uint32_t Count() const noexcept { return _count; }
    /// Compile phase only. Pointer-hash probe; never interns.
    UsdGenOpId Find(UsdGenToken const &name) const noexcept;
    UsdGenOpRecord const &Record(UsdGenOpId id) const noexcept { return _records[UsdGenIndex(id)]; }
    std::unique_ptr<UsdGenOp> Create(UsdGenOpId id) const;
    /// First manifest extension load failure; empty when all loaded.
    std::string const &InitError() const noexcept { return _initError; }
    /// The registered connected-parameter evaluator; create null when none.
    UsdGenParamEvaluatorDef const &ParamEvaluator() const noexcept { return _paramEvaluator; }

private:
    friend class UsdGenRegistrar;
    friend void UsdGenFreezeRegistries();
    friend UsdGenOpRegistry &UsdGenMutableOpRegistry() noexcept;
    friend UsdGenOpRegistry &UsdGenBuildOpRegistry(char const *manifest);
    explicit UsdGenOpRegistry(char const *manifest);
    UsdGenOpRecord _records[kUsdGenMaxOps];
    uint32_t _count = 0;
    struct NameSlot { UsdGenToken name; UsdGenOpId id = UsdGenOpId::Invalid; };
    NameSlot _byName[2 * kUsdGenMaxOps];
    UsdGenParamEvaluatorDef _paramEvaluator;
    std::string _initError;              // first extension load failure
    std::atomic<bool> _frozen{false};
};

/// Fills UsdGenOpNode::opId and runs every input record's intake hook (maps:
/// their type id). Compile phase.
USDGEN_CORE_API void UsdGenResolveDescIds(UsdGenGraphSpec *desc);

/// Generic expression admission over the record's decls. Messages (substrings tests check today):
///   "unsupported or incorrectly typed <displayName> expression <param>"
///   "<displayName> <param> requires groom evaluation"
///   "<displayName> <param> requires groom/primitive evaluation"
///   "operator <name> at <path> does not accept connected (expression) parameters"
/// 'spec' names the node in messages (its label).
USDGEN_CORE_API bool UsdGenValidateExpressionBindings(UsdGenOpRecord const &,
    UsdGenOpNode const &, UsdGenGraphSpec const *spec, std::vector<std::string> *errors);
/// The same for a node whose record is looked up (opId, else its type); an
/// unregistered type with bindings is rejected. Compile phase.
USDGEN_CORE_API bool UsdGenValidateExpressionBindings(UsdGenOpNode const &,
    UsdGenGraphSpec const *spec, std::vector<std::string> *errors);

}  // namespace usdGen
