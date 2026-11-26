// usdGen prim adapters (ADR §5.1 #3; 06-imaging.md §2).
//
// One adapter body behind the five plugInfo entries (UsdGenGroom/
// Description/Operator/Map/GuideSet derive types). primTypeName is a single
// string per plugInfo entry, and UsdGenGroom / UsdGenDescription /
// UsdGenGuideSet derive from UsdGeomImageable rather than UsdGenOperator, so
// includeDerivedPrimTypes cannot reach them from an operator base entry.
// Hence five one-line final subclasses of a single implementation.
//
// The subprim is ONE Hydra prim per USD prim: GetImagingSubprims returns
// {TfToken()} and the usdGen parameter container is published UNDER the
// prim-level data source at locator `usdGen` (06 §2.1/§2.3). Mappings are
// built generically from UsdPrimDefinition (R6: skip-list, never a
// name-prefix test) and cached once per schema type name.
#ifndef USDGEN_IMAGING_PRIM_ADAPTER_H
#define USDGEN_IMAGING_PRIM_ADAPTER_H

#include "usdGenImaging/api.h"

#include "pxr/usdImaging/usdImaging/dataSourceMapped.h"
#include "pxr/usdImaging/usdImaging/sceneIndexPrimAdapter.h"
#include "pxr/pxr.h"

PXR_NAMESPACE_OPEN_SCOPE

/// \class UsdGenPrimAdapterBase
///
/// Publishes every declared usdGen:* property of the prim as a data source
/// under the `usdGen` container (S14 pull-all), and maps each property edit
/// to its precise C1 locator so the UsdGenDirtyRouter gets exact dirties
/// (06 §2.7; 02-schema.md §6 is the property→locator table this emits).
class UsdGenPrimAdapterBase : public UsdImagingSceneIndexPrimAdapter
{
public:
    TfTokenVector GetImagingSubprims(UsdPrim const &prim) override;

    /// The empty token for all five registered types (06 §2.1): operator and
    /// map prims are data-only and must not become rprims; Groom/Description/
    /// GuideSet carry their Imageable state through
    /// UsdImagingDataSourcePrim, not through a Hydra type.
    TfToken GetImagingSubprimType(
            UsdPrim const &prim, TfToken const &subprim) override;

    HdContainerDataSourceHandle GetImagingSubprimData(
            UsdPrim const &prim,
            TfToken const &subprim,
            const UsdImagingDataSourceStageGlobals &stageGlobals) override;

    HdDataSourceLocatorSet InvalidateImagingSubprim(
            UsdPrim const &prim,
            TfToken const &subprim,
            TfTokenVector const &properties,
            UsdImagingPropertyInvalidationType invalidationType) override;

    /// The shared mapping table for one schema type name, built once from
    /// UsdPrimDefinition (06 §2.2). Owned by a static cache; safe to call
    /// from any thread (internal lock, released before use).
    ///
    /// CONCRETE TYPE ONLY: properties carried by an applied API schema are
    /// absent. Serving a prim needs the overload below, because a
    /// single-apply API is applied per prim and not by the type.
    static const UsdImagingDataSourceMapped::PropertyMappings &
        Mappings(TfToken const &schemaTypeName);

    /// The mapping table for a prim's FULL definition: its concrete type plus
    /// every applied API schema. `usdGen:look:*` lives on UsdGenLookAPI, which
    /// a scene applies per description, so a concrete-type-only table serves
    /// no `look` container at all and every look field silently reaches the
    /// engine as its schema default.
    static const UsdImagingDataSourceMapped::PropertyMappings &
        Mappings(UsdPrimTypeInfo const &typeInfo);

private:
    static const UsdImagingDataSourceMapped::PropertyMappings &
        _Mappings(TfToken const &key, UsdPrimDefinition const *definition);

public:

    /// `usdGen:clump:size` -> `clump/size` (the RELATIVE locator; Mappings
    /// makes it absolute under the `usdGen` prefix). Non-usdGen names are
    /// not expected (skip-list filtering); the name passes through.
    /// 3-arg form implements the locator contract (.omp/locator-
    /// contract.md §2 FINAL RULE): strict-ancestor properties take
    /// `-value`/`-rel` on the final element (per prim-type sibling set).
    static HdDataSourceLocator LocatorForProperty(
        TfToken const &property, bool isRelationship,
        TfTokenVector const &siblings);
    static HdDataSourceLocator LocatorForProperty(TfToken const &property);

    /// True for the two relationships 02-schema.md §2 declares "exactly one
    /// target"; every other usdGen relationship is array-mapped.
    static bool IsSingleTarget(TfToken const &property);
};

/// Five one-line final subclasses, one per concrete prim type that must be
/// individually claimed by the adapter registry (ADR §2.3).
class UsdGenGroomAdapter final : public UsdGenPrimAdapterBase {};
class UsdGenDescriptionAdapter final : public UsdGenPrimAdapterBase {};
class UsdGenOperatorAdapter final : public UsdGenPrimAdapterBase {};
class UsdGenMapAdapter final : public UsdGenPrimAdapterBase {};
class UsdGenGuideSetAdapter final : public UsdGenPrimAdapterBase {};
class UsdGenExpressionAdapter final : public UsdGenPrimAdapterBase {};

PXR_NAMESPACE_CLOSE_SCOPE

#endif // USDGEN_IMAGING_PRIM_ADAPTER_H
