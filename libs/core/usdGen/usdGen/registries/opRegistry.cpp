// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/registries/opRegistry.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGen/registries/opRecordTables.h"
#include "usdGen/registries/registrar.h"
#include "usdGen/registries/token.h"

#include "usdGen/api/graphSpec.h"
#include "usdGen/op.h"

#include <bitset>

namespace usdGen {

void UsdGenReadTokenCheckSetting() noexcept;   // token.cpp
UsdGenTypeRegistry &UsdGenMutableTypeRegistry() noexcept;

namespace {
std::atomic<UsdGenOpRegistry *> gOpRegistry{nullptr};
TfToken const kNoToken;
}  // namespace

// The interner asks whether interns are exec-phase without creating the
// registry (it interns while the registry is being built).
bool UsdGenOpRegistryFrozenIfBuilt() noexcept
{
    UsdGenOpRegistry const *r = gOpRegistry.load(std::memory_order_acquire);
    return r && r->IsFrozen();
}

// ---- UsdGenOpRecord ---------------------------------------------------------

UsdGenParamSlot UsdGenOpRecord::FindParam(TfToken const &name) const noexcept
{
    if (!_tables || name.IsEmpty()) return kUsdGenNoParam;
    Tables const &t = *_tables;
    for (size_t i = name.Hash() & t.mask;; i = (i + 1) & t.mask) {
        auto const &slot = t.hash[i];
        if (slot.first.IsEmpty()) return kUsdGenNoParam;
        if (slot.first == name) return slot.second;
    }
}

TfToken const &UsdGenOpRecord::ParamName(UsdGenParamSlot s) const noexcept
{
    return _tables && s < _tables->paramNames.size() ? _tables->paramNames[s] : kNoToken;
}

int UsdGenOpRecord::EnumIndex(UsdGenParamSlot s, TfToken const &value) const noexcept
{
    if (!_tables || s >= _tables->enumerants.size()) return -1;
    auto const &names = _tables->enumerants[s];
    for (size_t i = 0; i < names.size(); ++i)
        if (names[i] == value) return int(i);
    return -1;
}

int UsdGenOpRecord::EnumIndex(UsdGenParamSlot s, std::string const &value) const noexcept
{
    if (s >= _def.params.size()) return -1;
    TfSpan<char const *const> const names = _def.params[s].enumerants;
    for (size_t i = 0; i < names.size(); ++i)
        if (value == names[i]) return int(i);
    return -1;
}

template <class T>
static TfSpan<const T> SpanOf(std::vector<T> const &v) noexcept
{
    return TfSpan<const T>(v.data(), v.size());
}

TfSpan<const std::pair<TfToken, uint32_t>> UsdGenOpRecord::Routing() const noexcept
{
    return _tables ? SpanOf(_tables->routing) : TfSpan<const std::pair<TfToken, uint32_t>>();
}
TfSpan<const UsdGenParamSlot> UsdGenOpRecord::DigestSlots() const noexcept
{
    return _tables ? SpanOf(_tables->digest) : TfSpan<const UsdGenParamSlot>();
}
TfSpan<const TfToken> UsdGenOpRecord::ParamNames() const noexcept
{
    return _tables ? SpanOf(_tables->paramNames) : TfSpan<const TfToken>();
}
TfSpan<const TfToken> UsdGenOpRecord::ReferenceInputs() const noexcept
{
    return _tables ? SpanOf(_tables->referenceInputs) : TfSpan<const TfToken>();
}
TfSpan<const UsdGenPlane> UsdGenOpRecord::InputPlanes() const noexcept
{
    return _tables ? SpanOf(_tables->inputPlanes) : TfSpan<const UsdGenPlane>();
}
TfSpan<const UsdGenPlane> UsdGenOpRecord::OutputPlanes() const noexcept
{
    return _tables ? SpanOf(_tables->outputPlanes) : TfSpan<const UsdGenPlane>();
}
TfSpan<const TfToken> UsdGenOpRecord::MapRelationships() const noexcept
{
    return _tables ? SpanOf(_tables->mapRelationships) : TfSpan<const TfToken>();
}
TfToken const &UsdGenOpRecord::Token(uint16_t i) const noexcept
{
    return _tables && i < _tables->tokens.size() ? _tables->tokens[i] : kNoToken;
}

// ---- UsdGenOpRegistry -------------------------------------------------------

UsdGenOpRegistry::UsdGenOpRegistry()
{
    UsdGenReadTokenCheckSetting();
    (void)UsdGenTokens();
    (void)UsdGenMutableTypeRegistry();
    // The statically linked extensions, in id order.
    UsdGenRegistrar r(*this);
    usdGenRegister_usdGenCurves(r);
    usdGenRegister_usdGenExpr(r);
}

// Built on first use and published by CAS; a thread that loses the race
// discards its own copy, so nobody waits. Never freed.
UsdGenOpRegistry const &UsdGenOpRegistry::Get() noexcept
{
    return UsdGenMutableOpRegistry();
}

UsdGenOpRegistry &UsdGenMutableOpRegistry() noexcept
{
    if (UsdGenOpRegistry *registry = gOpRegistry.load(std::memory_order_acquire))
        return *registry;
    UsdGenOpRegistry *fresh = new UsdGenOpRegistry;
    UsdGenOpRegistry *winner = nullptr;
    if (gOpRegistry.compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                            std::memory_order_acquire))
        return *fresh;
    delete fresh;
    return *winner;
}

UsdGenOpId UsdGenOpRegistry::Find(TfToken const &name) const noexcept
{
    if (name.IsEmpty()) return UsdGenOpId::Invalid;
    constexpr size_t mask = 2 * kUsdGenMaxOps - 1;
    for (size_t i = name.Hash() & mask, n = 0; n <= mask; i = (i + 1) & mask, ++n) {
        NameSlot const &slot = _byName[i];
        if (slot.name.IsEmpty()) return UsdGenOpId::Invalid;
        if (slot.name == name) return slot.id;
    }
    return UsdGenOpId::Invalid;
}

std::unique_ptr<UsdGenOp> UsdGenOpRegistry::Create(UsdGenOpId id) const
{
    if (UsdGenIndex(id) >= _count) return nullptr;
    UsdGenOpRecord const &record = _records[UsdGenIndex(id)];
    return record.Def().create ? record.Def().create(record, record.Def().createContext)
                               : nullptr;
}

// ---- intake ------------------------------------------------------------------

void UsdGenResolveDescIds(UsdGenGraphSpec *desc)
{
    if (!desc) return;
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    UsdGenTypeRegistry const &types = UsdGenMutableTypeRegistry();
    for (UsdGenOpNode &node : desc->nodes) node.opId = ops.Find(node.type);
    for (UsdGenInputRecord &record : desc->inputs) {
        if (record.kind == UsdGenTypeId::Invalid) continue;
        UsdGenInputDesc const *input = types.Def(record.kind).input;
        if (input && input->intake) input->intake(record);
    }
}

// ---- expression admission ------------------------------------------------------

namespace {

std::string DomainList(expr::Domain domains)
{
    std::string text;
    auto add = [&](expr::Domain d, char const *name) {
        if ((uint8_t(domains) & uint8_t(d)) == 0) return;
        if (!text.empty()) text += '/';
        text += name;
    };
    add(expr::Domain::Groom, "groom");
    add(expr::Domain::Primitive, "primitive");
    add(expr::Domain::Point, "point");
    return text;
}

}  // namespace

bool UsdGenValidateExpressionBindings(UsdGenOpRecord const &record,
                                      UsdGenOpNode const &node,
                                      UsdGenGraphSpec const *spec,
                                      std::vector<std::string> *errors)
{
    if (node.expressionBindings.empty()) return true;
    auto reject = [errors](std::string message) {
        if (errors) errors->push_back(std::move(message));
        return false;
    };
    bool connectable = false;
    for (UsdGenParamDecl const &decl : record.Def().params)
        connectable = connectable || decl.expr.domains != expr::Domain::None;
    if (!connectable)
        return reject("operator " + record.Name().GetString() + " at " +
                      UsdGenLabel(spec, node.key) +
                      " does not accept connected (expression) parameters");

    char const *const display = record.Def().displayName ? record.Def().displayName
                                                         : record.Name().GetText();
    std::bitset<kUsdGenMaxParamsPerOp> seen;
    bool ok = true;
    for (UsdGenParamBinding const &binding : node.expressionBindings) {
        UsdGenParamSlot const slot = record.FindParam(binding.destination);
        std::string name = binding.destination.GetString();
        if (slot != kUsdGenNoParam) name = record.Def().params[slot].name;
        else if (name.compare(0, 7, "usdGen:") == 0) name.erase(0, 7);
        UsdGenExprAdmission const *admission =
            slot != kUsdGenNoParam ? &record.Def().params[slot].expr : nullptr;
        bool shapeOk = false;
        if (admission && admission->domains != expr::Domain::None && admission->nativeType) {
            auto const &shape = binding.destinationShape;
            shapeOk = !shape.isArray && shape.elementCount == 1 && shape.rows == 1 &&
                shape.columns == 1 && shape.components == admission->components &&
                shape.scalar == admission->scalar &&
                binding.nativeType.GetString() == admission->nativeType;
        }
        if (!shapeOk || seen.test(slot)) {
            ok = reject(std::string("unsupported or incorrectly typed ") + display +
                        " expression " + name);
            continue;
        }
        seen.set(slot);
        // A single-bit domain outside the admission fails; a groom-only
        // parameter also fails any other value (None, combined). Other
        // combined domains are left to the compiler's domain check.
        uint8_t const d = uint8_t(binding.domain), allowed = uint8_t(admission->domains);
        bool const single = d != 0 && (d & (d - 1)) == 0;
        bool const groomOnly = admission->domains == expr::Domain::Groom;
        if ((single && (d & allowed) == 0) ||
            (groomOnly && binding.domain != expr::Domain::Groom))
            ok = reject(std::string(display) + " " + name + " requires " +
                        DomainList(admission->domains) + " evaluation");
    }
    return ok;
}

bool UsdGenValidateExpressionBindings(UsdGenOpNode const &node,
                                      UsdGenGraphSpec const *spec,
                                      std::vector<std::string> *errors)
{
    if (node.expressionBindings.empty()) return true;
    UsdGenOpRegistry const &ops = UsdGenOpRegistry::Get();
    UsdGenOpId id = node.opId;
    if (id == UsdGenOpId::Invalid || UsdGenIndex(id) >= ops.Count() ||
        ops.Record(id).Name() != node.type)
        id = ops.Find(node.type);
    if (id == UsdGenOpId::Invalid) {
        if (errors)
            errors->push_back("operator " + node.type.GetString() + " at " +
                              UsdGenLabel(spec, node.key) +
                              " does not accept connected (expression) parameters");
        return false;
    }
    return UsdGenValidateExpressionBindings(ops.Record(id), node, spec, errors);
}

}  // namespace usdGen
