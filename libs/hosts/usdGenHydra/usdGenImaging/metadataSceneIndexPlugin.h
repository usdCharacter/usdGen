// Metadata-only UsdImagingSceneIndexPlugin for usdGen (ADR §5.1 #2, S7).
// Returns its input scene unchanged; exists solely to register the
// "usdGen" data-source name for instance aggregation and proxy-path
// translation.
#ifndef USDGEN_IMAGING_METADATA_SCENE_INDEX_PLUGIN_H
#define USDGEN_IMAGING_METADATA_SCENE_INDEX_PLUGIN_H

#include "usdGenImaging/api.h"
#include "usdGenImaging/usdGenEnable.h"
#include "pxr/usdImaging/usdImaging/sceneIndexPlugin.h"
#include "pxr/base/tf/token.h"

#include "pxr/pxr.h"

PXR_NAMESPACE_OPEN_SCOPE

class UsdGenMetadataSceneIndexPlugin final
    : public UsdImagingSceneIndexPlugin
{
public:
    HdSceneIndexBaseRefPtr
    AppendSceneIndex(
        HdSceneIndexBaseRefPtr const &inputScene) override
    {
        return inputScene;
    }

    // Must be non-const to match the base class virtual (ADR §5.1).
    // The shared kill switch (USDGEN_ENABLE) must also drop these names so
    // instance aggregation and proxy-path translation forget usdGen prims
    // when the plugin family is disabled (S-4).
    TfTokenVector
    InstanceDataSourceNames() override
    {
        return TfGetEnvSetting(USDGEN_ENABLE)
            ? TfTokenVector{ TfToken("usdGen") }
            : TfTokenVector();
    }

    TfTokenVector
    ProxyPathTranslationDataSourceNames() override
    {
        return TfGetEnvSetting(USDGEN_ENABLE)
            ? TfTokenVector{ TfToken("usdGen") }
            : TfTokenVector();
    }
};

PXR_NAMESPACE_CLOSE_SCOPE

#endif // USDGEN_IMAGING_METADATA_SCENE_INDEX_PLUGIN_H
