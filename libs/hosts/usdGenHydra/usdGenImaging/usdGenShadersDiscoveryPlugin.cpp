// usdGenShaders — SdrDiscoveryPlugin that discovers the usdGen hair preview
// glslfx shader defs from this plugin's resource directory.
//
// Compiled into libusdGenImaging.so. The usdGenShaders plugInfo.json points
// its LibraryPath at libusdGenImaging.so and its ShaderResources at
// "shaders/", so HioGlslfx can resolve the .glslfx files and Sdr can
// discover the three shader defs declared in shaders/shaderDefs.usda.
#include "usdGenImaging/usdGenShadersDiscoveryPlugin.h"
#include "usdGen/api/sharedCell.h"

#include "pxr/pxr.h"
#include "pxr/base/plug/plugin.h"
#include "pxr/base/plug/thisPlugin.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/base/tf/stringUtils.h"

#include "pxr/usd/ar/resolver.h"
#include "pxr/usd/ar/resolverContext.h"
#include "pxr/usd/ar/resolverContextBinder.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usdShade/shader.h"
#include "pxr/usd/usdShade/shaderDefUtils.h"

PXR_NAMESPACE_OPEN_SCOPE

namespace {

// The equivalent of static PlugPluginPtr plugin = PLUG_THIS_PLUGIN with
// PLUG_THIS_PLUGIN_NAME=usdGenShaders. We look up the plugin by name instead
// of defining the macro so the compile definition is not required on the
// whole target (plan/10 §4.4 alternative).
std::string
_GetShaderResourcePath(char const *resourceName = "")
{
    PlugPluginPtr const plugin =
        PlugRegistry::GetInstance().GetPluginWithName("usdGenShaders");
    const std::string path =
        PlugFindPluginResource(plugin, TfStringCatPaths("shaders",
                                                       resourceName));
    TF_VERIFY(!path.empty(), "Could not find shader resource: %s\n",
              resourceName);
    return path;
}

std::atomic<SdrStringVec const *> s_searchPaths{nullptr};

} // namespace

const SdrStringVec &
UsdGenShadersDiscoveryPlugin::GetSearchURIs() const
{
    // Installed once by CAS; never destroyed.
    return ::usdGen::UsdGenLazyLeaked(s_searchPaths,
        [] { return new SdrStringVec{_GetShaderResourcePath()}; });
}

SdrShaderNodeDiscoveryResultVec
UsdGenShadersDiscoveryPlugin::DiscoverShaderNodes(
    const SdrDiscoveryPluginContext & /*context*/)
{
    SdrShaderNodeDiscoveryResultVec result;

    std::string const shaderDefsFile = _GetShaderResourcePath("shaderDefs.usda");
    if (shaderDefsFile.empty()) {
        return result;
    }

    auto resolverContext =
        ArGetResolver().CreateDefaultContextForAsset(shaderDefsFile);

    const UsdStageRefPtr stage =
        UsdStage::Open(shaderDefsFile, resolverContext);
    if (!stage) {
        TF_RUNTIME_ERROR("Could not open file '%s' on a USD stage.",
                         shaderDefsFile.c_str());
        return result;
    }

    // SdfPath::GetText() returns const char* in 26.08 (no SdfString).

    ArResolverContextBinder binder(resolverContext);
    auto rootPrims = stage->GetPseudoRoot().GetChildren();
    for (const auto &shaderDef : rootPrims) {
        UsdShadeShader shader(shaderDef);
        if (!shader) {
            continue;
        }

        auto discoveryResults =
            UsdShadeShaderDefUtils::GetDiscoveryResults(
                shader, shaderDefsFile);

        result.insert(result.end(), discoveryResults.begin(),
                      discoveryResults.end());

        if (discoveryResults.empty()) {
            TF_RUNTIME_ERROR(
                "Found shader definition <%s> with no valid discovery "
                "results. This is likely because there are no resolvable "
                "info:sourceAsset values.",
                shaderDef.GetPath().GetText());
        }
    }

    return result;
}

SDR_REGISTER_DISCOVERY_PLUGIN(UsdGenShadersDiscoveryPlugin);

PXR_NAMESPACE_CLOSE_SCOPE
