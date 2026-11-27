// usdGenShaders — SdrDiscoveryPlugin that discovers the usdGen hair preview
// glslfx shaders from the usdGenShaders resource directory.
//
// Mirrors stock pxr/usd/plugin/usdShaders/discoveryPlugin.cpp:
//   - plugInfo: "Name": "usdGenShaders", "ShaderResources": "shaders"
//   - the plugin's resource dir holds shaderDefs.usda + the .glslfx files
//   - shaderDefs.usda defines the three Shader prims (UsdGenHairPreview,
//     UsdGenHairPreviewTranslucent, UsdGenHairPreviewPrimvar) which the
//     SdrGlslfxParserPlugin turns into Sdr shader nodes.
#ifndef USDGEN_IMAGING_SHADERS_DISCOVERY_PLUGIN_H
#define USDGEN_IMAGING_SHADERS_DISCOVERY_PLUGIN_H

#include "pxr/pxr.h"
#include "pxr/usd/sdr/discoveryPlugin.h"

PXR_NAMESPACE_OPEN_SCOPE

class UsdGenShadersDiscoveryPlugin final : public SdrDiscoveryPlugin
{
public:
    SdrShaderNodeDiscoveryResultVec
    DiscoverShaderNodes(const SdrDiscoveryPluginContext &context) override;

    // SdrDiscoveryPlugin::GetSearchURIs() is pure in 26.08 and returns a
    // const reference to a SdrStringVec — must be implemented verbatim.
    const SdrStringVec &GetSearchURIs() const override;
};

PXR_NAMESPACE_CLOSE_SCOPE

#endif  // USDGEN_IMAGING_SHADERS_DISCOVERY_PLUGIN_H
