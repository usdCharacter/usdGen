// Shared kill-switch for the usdGen scene index plugins (ADR §5.1).
// Both the renderer-level groom plugin and the UsdImaging metadata plugin
// consult USDGEN_ENABLE, so one switch disables all of usdGen's chain
// presence. Defined (TF_DEFINE_ENV_SETTING) in groomSceneIndexPlugin.cpp;
// declared here for the other translation units.
#ifndef USDGEN_IMAGING_ENABLE_H
#define USDGEN_IMAGING_ENABLE_H

#include "pxr/pxr.h"
#include "pxr/base/tf/envSetting.h"

PXR_NAMESPACE_OPEN_SCOPE

extern TfEnvSetting<bool> USDGEN_ENABLE;

PXR_NAMESPACE_CLOSE_SCOPE

#endif // USDGEN_IMAGING_ENABLE_H
