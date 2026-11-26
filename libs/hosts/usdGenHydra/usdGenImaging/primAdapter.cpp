// usdGen prim adapters (ADR §5.1 #3; 06-imaging.md §2).
//
// One adapter body under five plugInfo entries (06 §2.1): GetImagingSubprims
// returns {TfToken()} — one Hydra prim per USD prim — and
// GetImagingSubprimData overlays a single `usdGen` container (built by
// UsdImagingDataSourceMapped) onto UsdImagingDataSourcePrim.
//
// The mapping table is built GENERICALLY from UsdPrimDefinition (06 §2.2
// rules 1-4; R6: a skip-list, never a name-prefix test):
//   * every SdfSpecTypeAttribute  -> nested usdGen/<a>/<b> leaf
//   * every SdfSpecTypeRelationship -> usdGen/<name> path (array) or
//     single-path source, per the 02 §2.3 "exactly one target" set
//   * builtins already served by UsdImagingDataSourcePrim (visibility,
//     purpose, proxyPrim, xformOpOrder, xformOp:*, extent, primvars:*) are
//     the ONLY skipped names.
// A `float …:spline` ramp property (06 §2.4) is transported as the whole
// TsSpline via a custom factory and is never flagged time-varying.
//
// Invalidation (06 §2.7): 1:1 nested-locator invalidation from
// UsdImagingDataSourceMapped::Invalidate unioned with
// UsdImagingDataSourcePrim::Invalidate so the three Imageable types keep
// visibility/purpose/extent/primvars working. A resync of the bare `usdGen`
// container is emitted by the mapped source when a property appears or
// disappears; the dirty router treats that as graph-structural (02 §6.6
// rule 2).
#include "usdGenImaging/primAdapter.h"
#include "usdGen/sharedCell.h"

#include "usdGenUsd/walk/expressionConnection.h"

#include "usdGen/expressions/valueShape.h"

#include "pxr/base/ts/spline.h"
#include "pxr/imaging/hd/overlayContainerDataSource.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usdImaging/usdImaging/dataSourceMapped.h"
#include "pxr/usdImaging/usdImaging/dataSourcePrim.h"
#include "pxr/base/tf/hash.h"

#include <algorithm>
#include <string>
#include <vector>
#include <unordered_map>

#include "usdGenUsd/walk/usdGenTokens.h"
PXR_NAMESPACE_OPEN_SCOPE

namespace {

// A typed mapped source can suppress an incompatible authored property before
// the builder sees its VtValue. Preserve that error as independent metadata.
class _DedicatedTypeErrorsDataSource final
    : public HdTypedSampledDataSource<VtStringArray> {
public:
    HD_DECLARE_DATASOURCE(_DedicatedTypeErrorsDataSource);
    explicit _DedicatedTypeErrorsDataSource(UsdPrim const& prim) : _prim(prim) {}
    bool GetContributingSampleTimesForInterval(Time, Time, std::vector<Time>*) override { return false; }
    VtValue GetValue(Time t) override { return VtValue(GetTypedValue(t)); }
    VtStringArray GetTypedValue(Time) override {
        VtStringArray errors;
        const std::pair<char const*, char const*> fields[] = {
            {"usdGen:enabled", "bool"}, {"usdGen:seed", "int"},
            {"usdGen:width:default", "float"}, {"usdGen:tileTarget", "int"}};
        for (auto const& field : fields) {
            auto attr = _prim.GetAttribute(TfToken(field.first));
            VtValue value;
            if (!attr || !attr.Get(&value, UsdTimeCode::Default()) || value.IsEmpty()) continue;
            const std::string expected = field.second;
            const bool valid = expected == "bool" ? value.IsHolding<bool>() :
                expected == "float" ? value.IsHolding<float>() : value.IsHolding<int>();
            if (!valid)
                errors.push_back(_prim.GetPath().GetString() + ": " + field.first + " has wrong authored type");
        }
        return errors;
    }
private:
    UsdPrim _prim;
};

// Composed hierarchy is the stack topology.  USD's child range preserves the
// final composed order; visit it backwards and append an operator after its
// descendants.  Scopes deliberately contribute no entry.  Use the codeless
// base-schema query, not a concrete type-name list, so future operator types
// participate without adapter edits.
void
_AppendOperatorPostOrder(UsdPrim const &prim, VtArray<SdfPath> *out)
{
    std::vector<UsdPrim> children;
    for (UsdPrim const &child : prim.GetChildren()) children.push_back(child);
    for (auto child = children.rbegin(); child != children.rend(); ++child) {
        _AppendOperatorPostOrder(*child, out);
    }
    if (prim.IsA(TfToken("UsdGenOperator"))) {
        out->push_back(prim.GetPath());
    }
}

class _OperatorOrderDataSource final
    : public HdTypedSampledDataSource<VtArray<SdfPath>>
{
public:
    HD_DECLARE_DATASOURCE(_OperatorOrderDataSource);
    explicit _OperatorOrderDataSource(UsdPrim const &description)
        : _description(description) {}
    bool GetContributingSampleTimesForInterval(Time, Time,
                                                std::vector<Time> *) override
    { return false; }
    VtValue GetValue(Time t) override { return VtValue(GetTypedValue(t)); }
    VtArray<SdfPath> GetTypedValue(Time) override {
        VtArray<SdfPath> result;
        UsdPrim const ops = _description.GetChild(TfToken("Ops"));
        if (!ops) return result;
        // Bottom-up hierarchy, always: reverse prim order, post-order
        // over groups. Reordering the Ops children (USD spec order)
        // reorders execution; no metadata overrides it.
        std::vector<UsdPrim> children;
        for (UsdPrim const &child : ops.GetChildren()) children.push_back(child);
        for (auto child = children.rbegin(); child != children.rend(); ++child) {
            _AppendOperatorPostOrder(*child, &result);
        }
        return result;
    }
private:
    UsdPrim _description;
};

HdContainerDataSourceHandle
_OperatorOrderDataSource(UsdPrim const &description)
{
    return HdRetainedContainerDataSource::New(
        TfToken("operatorOrder"),
        _OperatorOrderDataSource::New(description));
}

HdSampledDataSourceHandle
_Value(VtValue const &value)
{
    return HdRetainedSampledDataSource::New(value);
}

void
_ExpressionArrayCount(UsdAttribute const& attr, uint32_t* count, bool* known)
{
    *count = 0;
    *known = !attr.GetTypeName().IsArray();
    if (*known) return;
    VtValue declaration;
    *known = attr.Get(&declaration) &&
        usdGen::expr::FixedArrayElementCount(attr.GetTypeName(), declaration, count);
}

// Dynamic expression metadata is not an authored schema property, so the
// mapped source cannot publish it.  Transport it as ordinary retained Hydra
// leaves.  Numeric child indices plus a `path` leaf are collision-free and
// preserve arbitrary USD names/paths without inventing an escaping grammar.
HdContainerDataSourceHandle
_ExpressionsDataSource(UsdPrim const &description)
{
    UsdPrim const root = description.GetChild(TfToken("Expressions"));
    TfTokenVector names;
    std::vector<HdDataSourceBaseHandle> values;
    size_t index = 0;
    if (!root) return HdRetainedContainerDataSource::New();
    for (UsdPrim const &expr : root.GetChildren()) {
        if (expr.GetPrimTypeInfo().GetTypeName() != TfToken("UsdGenExpression")) continue;
        TfTokenVector fields;
        std::vector<HdDataSourceBaseHandle> fieldValues;
        fields.push_back(TfToken("path")); fieldValues.push_back(_Value(VtValue(expr.GetPath())));
        std::string source;
        if (UsdAttribute a = expr.GetAttribute(TfToken("usdGen:expr:source"))) a.Get(&source);
        fields.push_back(TfToken("source")); fieldValues.push_back(_Value(VtValue(source)));
        TfTokenVector outputNames;
        std::vector<HdDataSourceBaseHandle> outputValues;
        size_t outputIndex = 0;
        for (UsdAttribute const &a : expr.GetAttributes()) {
            std::string const n = a.GetName().GetString();
            if (n.rfind("outputs:", 0) != 0) continue;
            uint32_t arrayElementCount = 0;
            bool arrayCountKnown = false;
            _ExpressionArrayCount(a, &arrayElementCount, &arrayCountKnown);
            TfTokenVector outFields{TfToken("name"), TfToken("nativeType"), TfToken("shape"),
                TfToken("arrayElementCount"), TfToken("arrayCountKnown")};
            std::vector<HdDataSourceBaseHandle> outValues{
                _Value(VtValue(TfToken(n.substr(8)))),
                _Value(VtValue(a.GetTypeName().GetAsToken())),
                _Value(VtValue(a.GetTypeName().GetAsToken())),
                _Value(VtValue(arrayElementCount)), _Value(VtValue(arrayCountKnown))};
            outputNames.push_back(TfToken(std::to_string(outputIndex++)));
            outputValues.push_back(HdRetainedContainerDataSource::New(
                outFields.size(), outFields.data(), outValues.data()));
        }
        fields.push_back(TfToken("outputs"));
        fieldValues.push_back(HdRetainedContainerDataSource::New(
            outputNames.size(), outputNames.data(), outputValues.data()));
        // input:<name> relationships: the data geoSampler("<name>") and
        // ptex("<name>") read. Forwarded targets, so a relationship that
        // targets another relationship resolves like a connection would.
        TfTokenVector inputNames;
        std::vector<HdDataSourceBaseHandle> inputValues;
        size_t inputIndex = 0;
        for (UsdRelationship const &rel : expr.GetRelationships()) {
            std::string const n = rel.GetName().GetString();
            if (n.rfind("input:", 0) != 0 || n.size() <= 6) continue;
            SdfPathVector targets;
            rel.GetForwardedTargets(&targets);
            TfTokenVector inFields{TfToken("name"), TfToken("targets")};
            std::vector<HdDataSourceBaseHandle> inValues{
                _Value(VtValue(TfToken(n.substr(6)))),
                _Value(VtValue(VtArray<SdfPath>(targets.begin(), targets.end())))};
            inputNames.push_back(TfToken(std::to_string(inputIndex++)));
            inputValues.push_back(HdRetainedContainerDataSource::New(
                inFields.size(), inFields.data(), inValues.data()));
        }
        fields.push_back(TfToken("inputs"));
        fieldValues.push_back(HdRetainedContainerDataSource::New(
            inputNames.size(), inputNames.data(), inputValues.data()));
        names.push_back(TfToken(std::to_string(index++)));
        values.push_back(HdRetainedContainerDataSource::New(
            fields.size(), fields.data(), fieldValues.data()));
    }
    return HdRetainedContainerDataSource::New(names.size(), names.data(), values.data());
}

HdContainerDataSourceHandle
_ExpressionBindingsDataSource(UsdPrim const &prim)
{
    TfTokenVector names;
    std::vector<HdDataSourceBaseHandle> values;
    size_t index = 0;
    for (UsdAttribute const &attr : prim.GetAttributes()) {
        SdfPathVector connections;
        attr.GetConnections(&connections);
        for (SdfPath const &connection : connections) {
            SdfPath const exprPath = connection.GetPrimPath();
            // A connection may name the expression prim instead of one of its
            // outputs; resolve both spellings to the same binding.  Preserve
            // malformed connections too: the compiler owns the fail-closed
            // diagnostic, and dropping one here would silently turn a
            // connected parameter back into its literal.
            TfToken const output =
                ::usdGenImaging::UsdGenResolveExpressionConnectionOutput(
                    prim, connection);
            TfToken domain("groom");
            VtValue cd = attr.GetCustomDataByKey(TfToken("usdGen:evaluation"));
            if (cd.IsHolding<std::string>()) domain = TfToken(cd.UncheckedGet<std::string>());
            VtValue literal; attr.Get(&literal);
            TfTokenVector fields{TfToken("expression"), TfToken("output"),
                TfToken("destination"), TfToken("nativeType"), TfToken("shape"),
                TfToken("domain"), TfToken("literal")};
            std::vector<HdDataSourceBaseHandle> fieldValues{
                _Value(VtValue(exprPath)), _Value(VtValue(output)),
                _Value(VtValue(attr.GetName())), _Value(VtValue(attr.GetTypeName().GetAsToken())),
                _Value(VtValue(attr.GetTypeName().GetAsToken())), _Value(VtValue(domain)), _Value(literal)};
            names.push_back(TfToken(std::to_string(index++)));
            values.push_back(HdRetainedContainerDataSource::New(
                fields.size(), fields.data(), fieldValues.data()));
        }
    }
    return HdRetainedContainerDataSource::New(names.size(), names.data(), values.data());
}

// The stage scene index retains adapter data-source handles across edits.
// Keep expression snapshots live at the container boundary so a source,
// output, connection, or evaluation metadata edit is visible without
// rebuilding the scene-index chain.
class _LiveExpressionsDataSource final : public HdContainerDataSource
{
public:
    HD_DECLARE_DATASOURCE(_LiveExpressionsDataSource);
    explicit _LiveExpressionsDataSource(UsdPrim const &prim) : _prim(prim) {}
    TfTokenVector GetNames() override { return _ExpressionsDataSource(_prim)->GetNames(); }
    HdDataSourceBaseHandle Get(TfToken const &name) override
    { return _ExpressionsDataSource(_prim)->Get(name); }
private: UsdPrim _prim;
};

class _LiveExpressionBindingsDataSource final : public HdContainerDataSource
{
public:
    HD_DECLARE_DATASOURCE(_LiveExpressionBindingsDataSource);
    explicit _LiveExpressionBindingsDataSource(UsdPrim const &prim) : _prim(prim) {}
    TfTokenVector GetNames() override { return _ExpressionBindingsDataSource(_prim)->GetNames(); }
    HdDataSourceBaseHandle Get(TfToken const &name) override
    { return _ExpressionBindingsDataSource(_prim)->Get(name); }
private: UsdPrim _prim;
};

// Stage metadata is not a prim property, so it cannot be transported by the
// generic mapped source.  Keep this description-level source live: a retained
// Hydra handle re-reads the composed stage metadata after an edit rather than
// freezing the first frame's rate.
class _TimeCodesPerSecondDataSource final : public HdSampledDataSource
{
public:
    HD_DECLARE_DATASOURCE(_TimeCodesPerSecondDataSource);
    explicit _TimeCodesPerSecondDataSource(UsdPrim const &prim) : _prim(prim) {}

    VtValue GetValue(Time) override
    {
        VtValue value;
        UsdStageRefPtr const stage = _prim ? _prim.GetStage() : nullptr;
        if (!stage) return VtValue();

        // Preserve an authored malformed value for the builder's strict
        // validation.  Otherwise use USD's resolved accessor, which applies
        // the documented timeCodesPerSecond/session/root, then
        // framesPerSecond/session/root, then 24 fallback precedence.
        if (stage->HasAuthoredMetadata(TfToken("timeCodesPerSecond"))) {
            if (stage->GetMetadata(TfToken("timeCodesPerSecond"), &value)) {
                return value;
            }
            return VtValue();
        }
        return VtValue(stage->GetTimeCodesPerSecond());
    }

    bool GetContributingSampleTimesForInterval(
        Time, Time, std::vector<Time> *) override
    { return false; }

private:
    UsdPrim _prim;
};

class _LiveDescriptionRuntimeDataSource final : public HdContainerDataSource
{
public:
    HD_DECLARE_DATASOURCE(_LiveDescriptionRuntimeDataSource);
    explicit _LiveDescriptionRuntimeDataSource(UsdPrim const &prim)
        : _prim(prim) {}

    TfTokenVector GetNames() override
    { return TfTokenVector{TfToken("timeCodesPerSecond")}; }

    HdDataSourceBaseHandle Get(TfToken const &name) override
    {
        if (name != TfToken("timeCodesPerSecond")) return nullptr;
        return _TimeCodesPerSecondDataSource::New(_prim);
    }

private:
    UsdPrim _prim;
};

// 06 §2.2 rule 3: the skip-list is exactly what
bool
_IsPrimBuiltin(TfToken const &name)
{
    // 06 §2.2 rule 3: skip ONLY what UsdImagingDataSourcePrim already serves.
    if (name == TfToken("visibility") || name == TfToken("purpose") ||
        name == TfToken("proxyPrim") || name == TfToken("xformOpOrder") ||
        name == TfToken("extent")) {
        return true;
    }
    const std::string s = name.GetString();
    return s.rfind("xformOp:", 0) == 0 || s.rfind("primvars:", 0) == 0;
}

// 06 §2.4 / S11: whole-spline ramp transport. Reads the authored spline and
// returns a retained TsSpline source WITHOUT flagging time-varying (a .spline
// on a plain float would otherwise dirty on every SetTime — MEASURED probe3).
HdSampledDataSourceHandle
_SplineRampFactory(
    UsdAttribute const &attr,
    UsdImagingDataSourceStageGlobals const &,
    SdfPath const &,
    HdDataSourceLocator const &)
{
    TsSpline spline = attr.GetSpline();
    if (spline.IsEmpty()) {
        return nullptr;  // no spline authored: no opinion at this leaf
    }
    return HdRetainedTypedSampledDataSource<TsSpline>::New(spline);
}

}  // namespace

TfTokenVector
UsdGenPrimAdapterBase::GetImagingSubprims(UsdPrim const &)
{
    // 06 §2.1: one Hydra prim per USD prim, no subprims.
    return TfTokenVector{ TfToken() };
}

TfToken
UsdGenPrimAdapterBase::GetImagingSubprimType(
        UsdPrim const &, TfToken const &)
{
    return TfToken();
}

HdContainerDataSourceHandle
UsdGenPrimAdapterBase::GetImagingSubprimData(
        UsdPrim const &prim,
        TfToken const &subprim,
        const UsdImagingDataSourceStageGlobals &stageGlobals)
{
    if (subprim != TfToken()) {
        return nullptr;
    }
    HdContainerDataSourceHandle base =
        UsdImagingDataSourcePrim::New(
            prim.GetPath(), prim, stageGlobals);
    HdContainerDataSourceHandle usdGen =
        UsdImagingDataSourceMapped::New(
            prim, prim.GetPath(),
            Mappings(prim.GetPrimTypeInfo()),
            stageGlobals);
    if (!usdGen) {
        return base;
    }
    if (prim.GetPrimTypeInfo().GetTypeName() == TfToken("UsdGenDescription")) {
        // This sits above the mapped root because it is derived composition
        // state, rather than an authored usdGen property.
        usdGen = HdOverlayContainerDataSource::OverlayedContainerDataSources(
            _OperatorOrderDataSource(prim), usdGen);
        usdGen = HdOverlayContainerDataSource::OverlayedContainerDataSources(
            HdRetainedContainerDataSource::New(TfToken("expressions"),
                                               _LiveExpressionsDataSource::New(prim)), usdGen);
        usdGen = HdOverlayContainerDataSource::OverlayedContainerDataSources(
            HdRetainedContainerDataSource::New(
                TfToken("usdGenRuntime"),
                _LiveDescriptionRuntimeDataSource::New(prim)), usdGen);
    } else if (prim.IsA(TfToken("UsdGenOperator"))) {
        usdGen = HdOverlayContainerDataSource::OverlayedContainerDataSources(
            HdRetainedContainerDataSource::New(TfToken("expressionBindings"),
                                               _LiveExpressionBindingsDataSource::New(prim)), usdGen);
    }
    // The usdGen container and the prim-level containers are disjoint
    // locators; the overlay exists so downstream flattening sees one root.
    auto combined = HdOverlayContainerDataSource::OverlayedContainerDataSources(usdGen, base);
    return HdOverlayContainerDataSource::OverlayedContainerDataSources(
        HdRetainedContainerDataSource::New(TfToken("__usdGenValidationErrors"),
                                          _DedicatedTypeErrorsDataSource::New(prim)), combined);
}

HdDataSourceLocatorSet
UsdGenPrimAdapterBase::InvalidateImagingSubprim(
        UsdPrim const &prim,
        TfToken const &subprim,
        TfTokenVector const &properties,
        UsdImagingPropertyInvalidationType invalidationType)
{
    // 06 §2.7: the mapped source gives 1:1 nested-locator invalidation for
    // every usdGen:* property edit (usdGen:noise:magnitude ->
    // usdGen/noise/magnitude); the union keeps visibility/purpose/extent/
    // primvars working on the Imageable types. Taken unconditionally: a
    // dirty for a locator that is never published is ignored by Hydra.
    HdDataSourceLocatorSet result =
        UsdImagingDataSourceMapped::Invalidate(
            properties,
            Mappings(prim.GetPrimTypeInfo()));
    result.insert(
        UsdImagingDataSourcePrim::Invalidate(
            prim, subprim, properties, invalidationType));
    for (auto const& property : properties) {
        if (property == TfToken("usdGen:enabled") || property == TfToken("usdGen:seed") ||
            property == TfToken("usdGen:width:default") || property == TfToken("usdGen:tileTarget")) {
            result.insert(HdDataSourceLocator(TfToken("__usdGenValidationErrors")));
            break;
        }
    }
    if (prim.GetPrimTypeInfo().GetTypeName() == TfToken("UsdGenDescription")) {
        // Child-order metadata produces a resync on the Description.  The
        // groom scene index also forwards descendant add/remove/reorder to
        // this locator; accepting every Description invalidation here keeps
        // the aggregate coherent for direct USD notice delivery.
        result.insert(HdDataSourceLocator(
            PXR_NS::usdGenImaging::UsdGenContainerToken()).Append(
                TfToken("operatorOrder")));
        result.insert(HdDataSourceLocator(
            PXR_NS::usdGenImaging::UsdGenContainerToken()).Append(
                TfToken("expressions")));
    } else if (prim.IsA(TfToken("UsdGenOperator"))) {
        result.insert(HdDataSourceLocator(
            PXR_NS::usdGenImaging::UsdGenContainerToken()).Append(
                TfToken("expressionBindings")));
    } else if (prim.GetPrimTypeInfo().GetTypeName() == TfToken("UsdGenExpression")) {
        // input:<name> relationships and outputs:* are dynamic, so the mapped
        // table knows nothing of them. Dirty the expression prim for them all
        // the same: the owning description re-reads its live expression
        // aggregate on the resulting notice.
        for (auto const& property : properties) {
            std::string const n = property.GetString();
            if (n.rfind("input:", 0) == 0 || n.rfind("outputs:", 0) == 0) {
                result.insert(HdDataSourceLocator(
                    PXR_NS::usdGenImaging::UsdGenContainerToken()).Append(
                        TfToken("inputs")));
                break;
            }
        }
    }
    return result;
}

const UsdImagingDataSourceMapped::PropertyMappings &
UsdGenPrimAdapterBase::Mappings(UsdPrimTypeInfo const &typeInfo)
{
    // Applied API schemas are applied per PRIM, so the table has to be keyed
    // by the prim's whole definition, not by its type name. Without the
    // applied set in the key, two descriptions of the same type -- one with
    // UsdGenLookAPI, one without -- would share the first table built.
    std::string key = typeInfo.GetSchemaTypeName().GetString();
    for (TfToken const &api : typeInfo.GetAppliedAPISchemas()) {
        key += '|';
        key += api.GetString();
    }
    return _Mappings(TfToken(key), &typeInfo.GetPrimDefinition());
}

const UsdImagingDataSourceMapped::PropertyMappings &
UsdGenPrimAdapterBase::Mappings(TfToken const &schemaTypeName)
{
    // FindConcretePrimDefinition returns nullptr for abstract types and for
    // names with no registered schema — such a prim has no declared
    // usdGen:* properties, so the cached (empty) table is correct (06 §2.2
    // rule 2's null branch).
    return _Mappings(schemaTypeName,
                     UsdSchemaRegistry::GetInstance()
                         .FindConcretePrimDefinition(schemaTypeName));
}

namespace {
// Append-only table of immutable mappings, copy-on-write by CAS. Every later
// table holds each mapping too, so a returned reference never dangles.
using _MappingTable = std::unordered_map<
    TfToken, std::shared_ptr<const UsdImagingDataSourceMapped::PropertyMappings>, TfHash>;
// Leaked: data sources may keep a returned mapping past static destruction.
std::atomic<::usdGen::UsdGenSharedCell<_MappingTable const> *> s_mappingTable{nullptr};
::usdGen::UsdGenSharedCell<_MappingTable const> &MappingTable()
{
    return ::usdGen::UsdGenLazyLeaked(s_mappingTable);
}
}

const UsdImagingDataSourceMapped::PropertyMappings &
UsdGenPrimAdapterBase::_Mappings(TfToken const &key, UsdPrimDefinition const *def)
{
    using Mapping = UsdImagingDataSourceMapped::PropertyMappings;
    if (auto table = MappingTable().Load()) {
        if (auto it = table->find(key); it != table->end()) return *it->second;
    }
    std::vector<UsdImagingDataSourceMapped::PropertyMapping> mappings;
    if (def) {
            // Sibling set for the ancestor pass (contract S3.1): the whole
            // definition is in hand, so the full property set is too — no new
            // USD calls.
            TfTokenVector const &siblings = def->GetPropertyNames();
            for (TfToken const &name : siblings) {
                if (_IsPrimBuiltin(name)) {
                    continue;
                }
                switch (def->GetSpecType(name)) {
                case SdfSpecTypeAttribute:
                    {
                        UsdImagingDataSourceMapped::AttributeMapping m;
                        m.usdName = name;
                        m.hdLocator = LocatorForProperty(
                            name, /*isRelationship=*/false, siblings);
                        if (const std::string ps = name.GetString();
                            ps.size() >= 7 && ps.compare(ps.size() - 7, 7, ":spline") == 0) {
                            // Whole-spline ramp transport (06 §2.4): the stock
                            // factory would flag a .spline time-varying.
                            m.factory = _SplineRampFactory;
                        }
                        mappings.push_back(std::move(m));
                    }
                    break;
            case SdfSpecTypeRelationship:
                {
                    UsdImagingDataSourceMapped::RelationshipMapping m;
                    m.usdName = name;
                    m.hdLocator = LocatorForProperty(
                        name, /*isRelationship=*/true, siblings);
                    m.factory = IsSingleTarget(name)
                        ? UsdImagingDataSourceMapped::GetPathFromRelationshipDataSourceFactory()
                        : UsdImagingDataSourceMapped::GetPathArrayFromRelationshipDataSourceFactory();
                    mappings.push_back(std::move(m));
                }
                break;
            default:
                break;
            }
        }
    }
    auto slot = std::make_shared<const Mapping>(
        mappings, HdDataSourceLocator(PXR_NS::usdGenImaging::UsdGenContainerToken()));
    // Concurrent builders may construct equivalent candidates.  Only the
    // canonical inserted value is returned, so all callers retain a stable
    // reference and no mutable published object is ever replaced.
    ::usdGen::UsdGenCowUpdateIf(MappingTable(), [&](_MappingTable &table) {
        return table.emplace(key, slot).second;
    });
    return *MappingTable().Load()->at(key);
}

// Split per 02 §0.7 (relative elements, no container prefix).
static std::vector<std::string>
_Split02(TfToken const &property)
{
    const std::string s = property.GetString();
    const size_t prefix = std::string("usdGen:").size();
    const std::string rest =
        (s.size() >= prefix && s.compare(0, prefix, "usdGen:") == 0)
            ? s.substr(prefix)
            : s;
    std::vector<std::string> elements;
    size_t begin = 0;
    while (begin <= rest.size()) {
        const size_t pos = rest.find(':', begin);
        elements.push_back(
            pos == std::string::npos
                ? rest.substr(begin)
                : rest.substr(begin, pos - begin));
        if (pos == std::string::npos) {
            break;
        }
        begin = pos + 1;
    }
    if (elements.empty()) {
        elements.push_back(s);
    }
    return elements;
}

HdDataSourceLocator
UsdGenPrimAdapterBase::LocatorForProperty(TfToken const &property,
                                         bool isRelationship,
                                         TfTokenVector const &siblings)
{
    // 06 §2.2 rule 4 as amended by the locator contract (.omp/locator-
    // contract.md §2 FINAL RULE): base rule unchanged (02 §0.7 — strip
    // "usdGen:", split rest on ':' → clump/size, frozen/mode); Mappings()
    // prepends the `usdGen` container prefix. DEVIATION & REASON: when this
    // property's base locator is a STRICT ancestor of any sibling's base
    // locator on the same prim type, the ancestor's FINAL element takes
    // "-value" (attribute) / "-rel" (relationship). Without it, an
    // ancestor/descendant pair (usdGen:length:value vs usdGen:length:mode)
    // registers a leaf over a container node and UsdImagingDataSourceMapped
    // TF_CODING_ERRORs (dataSourceMapped.cpp:272 "already an ascendant
    // locator") and drops the container. Descendants NEVER move (R25
    // longest-prefix matching preserved); non-colliding properties keep
    // 02 §0.7 locators exactly (all 02 §6.1 rel rows unchanged).
    // Pure: same (property, isRelationship, siblings) -> same locator.
    std::vector<std::string> elements = _Split02(property);
    for (TfToken const &sib : siblings) {
        if (sib == property) continue;
        std::vector<std::string> sels = _Split02(sib);
        if (elements.size() < sels.size() &&
            std::equal(elements.begin(), elements.end(), sels.begin())) {
            elements.back() += isRelationship ? "-rel" : "-value";
            break;  // one suffix suffices: suffixing only ever extends the
                    // ancestor side, so no two suffixed forms collide.
        }
    }
    TfTokenVector tokens;
    tokens.reserve(elements.size());
    for (std::string const &e : elements) tokens.push_back(TfToken(e));
    return HdDataSourceLocator(tokens.size(), tokens.data());
}

HdDataSourceLocator
UsdGenPrimAdapterBase::LocatorForProperty(TfToken const &property)
{
    // Single-property form: no sibling set, so no ancestor check. Used by
    // test/diagnostic paths that resolve one name outside a type context;
    // Mappings() always uses the 3-arg form above.
    std::vector<std::string> elements = _Split02(property);
    TfTokenVector tokens;
    tokens.reserve(elements.size());
    for (std::string const &e : elements) tokens.push_back(TfToken(e));
    return HdDataSourceLocator(tokens.size(), tokens.data());
}

bool
UsdGenPrimAdapterBase::IsSingleTarget(TfToken const &property)
{
    // 06 §2.3: exactly one target. Every other relationship keeps the array
    // factory — an empty array is how the evaluator sees "not bound".
    return property == TfToken("usdGen:frozen:curves");
}

#define USDGEN_DEFINE_PRIM_ADAPTER(AdapterType)                              \
    TF_REGISTRY_FUNCTION(TfType) {                                           \
        TfType t = TfType::Define<                                           \
            AdapterType,                                                    \
            TfType::Bases<UsdImagingSceneIndexPrimAdapter>>();               \
        t.SetFactory<UsdImagingPrimAdapterFactory<AdapterType>>();           \
    }

USDGEN_DEFINE_PRIM_ADAPTER(UsdGenGroomAdapter)
USDGEN_DEFINE_PRIM_ADAPTER(UsdGenDescriptionAdapter)
USDGEN_DEFINE_PRIM_ADAPTER(UsdGenOperatorAdapter)
USDGEN_DEFINE_PRIM_ADAPTER(UsdGenMapAdapter)
USDGEN_DEFINE_PRIM_ADAPTER(UsdGenGuideSetAdapter)
USDGEN_DEFINE_PRIM_ADAPTER(UsdGenExpressionAdapter)

#undef USDGEN_DEFINE_PRIM_ADAPTER

PXR_NAMESPACE_CLOSE_SCOPE
