// usdGenMetadataSceneIndexPlugin: metadata-only UsdImagingSceneIndexPlugin (S7).
//
// Returns its input unchanged; it exists solely to register the "usdGen"
// instance-data-source and proxy-path-translation names so instance
// aggregation and proxy-path translation account for usdGen prims.
//
// Registration note (S-4): the type is defined exactly once, via
// UsdImagingSceneIndexPlugin::Define<T>() inside the
// TF_REGISTRY_FUNCTION(UsdImagingSceneIndexPlugin) block below. That helper
// already performs TfType::Define<T, Bases<UsdImagingSceneIndexPlugin>>() and
// installs the UsdImaging factory — an extra explicit TF_REGISTRY_FUNCTION(TfType)
// block would redefine the same C++ type and emit the "TfType ... already has
// a defined C++ type" coding error.
#include "usdGenImaging/metadataSceneIndexPlugin.h"

#include "pxr/usdImaging/usdImaging/sceneIndexPlugin.h"

PXR_NAMESPACE_OPEN_SCOPE

TF_REGISTRY_FUNCTION(UsdImagingSceneIndexPlugin)
{
    UsdImagingSceneIndexPlugin::Define<UsdGenMetadataSceneIndexPlugin>();
}

PXR_NAMESPACE_CLOSE_SCOPE
