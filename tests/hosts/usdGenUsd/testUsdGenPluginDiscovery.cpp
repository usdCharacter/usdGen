// testUsdGenPluginDiscovery (T1): verify the plugins load and their types
// are registered in the TfType registry, codeless schema types resolve, and
// the codeless UsdGenDescription reports a non-empty computed extent.
//
// Usage: testUsdGenPluginDiscovery <schemaResourcesDir> <imagingResourcesDir> <fixture.usda>
#include "utils/pxrInterop.h"
#include "usdGenTestUtilsHd/testUtilsHd.h"

#include "pxr/pxr.h"
#include "pxr/usd/sdf/path.h"
#include "pxr/base/plug/registry.h"
#include "pxr/base/tf/token.h"
#include "pxr/base/tf/type.h"
#include "pxr/base/tf/getenv.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/vt/value.h"
#include "pxr/imaging/hd/sceneIndexPlugin.h"
#include "pxr/usdImaging/usdImaging/sceneIndexPlugin.h"
#include "pxr/usd/usd/prim.h"
#include "pxr/usd/usd/schemaRegistry.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usd/usd/timeCode.h"
#include "pxr/usd/usdGeom/boundable.h"

#include "pxr/base/plug/plugin.h"
#include "pxr/base/plug/registry.h"
#include <cstdio>
#include <string>


PXR_NAMESPACE_USING_DIRECTIVE

namespace {
int g_failures = 0;
void Check(bool ok, const char *what) {
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what); }
    else      { std::printf("ok:   %s\n", what); }
}

bool TypeKnown(const char *name)
{
    const TfType t = TfType::FindByName(name);
    return !t.IsUnknown();
}

// Platform-neutral replacement for POSIX ::access(F_OK).
bool FileExists(const std::string &path)
{
    if (FILE *f = std::fopen(path.c_str(), "rb")) {
        std::fclose(f);
        return true;
    }
    return false;
}

}  // namespace

int main(int argc, char **argv)
{
    std::printf("=== testUsdGenPluginDiscovery ===\n");
    if (argc != 4) {
        std::printf(
            "usage: %s <schemaResourcesDir> <imagingResourcesDir> <fixture.usda>\n",
            argv[0]);
        return 2;
    }
    const std::string schemaResources = argv[1];
    const std::string imagingResources = argv[2];
    const std::string fixture = argv[3];

    const char *pluginPath = std::getenv("PXR_PLUGINPATH_NAME");
    Check(pluginPath != nullptr, "PXR_PLUGINPATH_NAME is set");
    if (pluginPath) std::printf("  PXR_PLUGINPATH_NAME = %s\n", pluginPath);

    // Register the build-tree plugin resource dirs explicitly (the ctest
    // ENVIRONMENT already puts them on PXR_PLUGINPATH_NAME; this is a belt).
    PlugRegistry::GetInstance().RegisterPlugins({schemaResources, imagingResources});
    // Plug registry discovery: both plugins must resolve by name from the
    // resource dirs registered above — this is what SdrCatalog does at runtime,
    // so a missing or malformed plugInfo.json fails loudly here instead of as
    // an unexplained "type not found" below.
    const auto &registry = PlugRegistry::GetInstance();
    for (const char *pluginName : {"usdGenSchema", "usdGenImaging"}) {
        const PlugPlugin *plugin = get_pointer(registry.GetPluginWithName(pluginName));
        if (!plugin) {
            std::printf("FAIL: plugin '%s' not found in PlugRegistry after registering %s and %s — "
                        "check that <dir>/plugInfo.json exists, is valid JSON, and names a loadable library\n",
                        pluginName, schemaResources.c_str(), imagingResources.c_str());
            ++g_failures;
        } else {
            const std::string libPath = plugin->GetPath();
            Check(!libPath.empty() && FileExists(libPath),
                  (std::string("plugin '") + pluginName + "' library exists: " + libPath).c_str());
        }
    }

    // TfType registry: imaging plugin types (registered via TF_REGISTRY_FUNCTION
    // inside libusdGenImaging.so, which is linked into this test binary).
    Check(TypeKnown("UsdGenGroomSceneIndexPlugin"),
          "UsdGenGroomSceneIndexPlugin type registered");
    if (TypeKnown("UsdGenGroomSceneIndexPlugin")) {
        const TfType groom = TfType::FindByName("UsdGenGroomSceneIndexPlugin");
        const TfType base = TfType::FindByName("HdSceneIndexPlugin");
        Check(groom.IsA(base),
              "UsdGenGroomSceneIndexPlugin derives from HdSceneIndexPlugin");
    }
    Check(TypeKnown("UsdGenMetadataSceneIndexPlugin"),
          "UsdGenMetadataSceneIndexPlugin type registered");

    // Codeless schema types, registered by the usdGenSchema plugin.
    Check(TypeKnown("UsdGenGroom"), "UsdGenGroom type known");
    Check(TypeKnown("UsdGenDescription"), "UsdGenDescription type known");
    Check(TypeKnown("UsdGenOperator"), "UsdGenOperator type known");
    Check(TypeKnown("UsdGenScatter"), "UsdGenScatter type known");
    Check(TypeKnown("UsdGenLookAPI"), "UsdGenLookAPI type known");

    // UsdSchemaRegistry view.
    Check(UsdSchemaRegistry::IsConcrete(TfToken("UsdGenScatter")),
          "UsdSchemaRegistry: UsdGenScatter is concrete");
    Check(UsdSchemaRegistry::IsConcrete(TfToken("UsdGenDescription")),
          "UsdSchemaRegistry: UsdGenDescription is concrete");
    Check(UsdSchemaRegistry::IsAbstract(TfToken("UsdGenOperator")),
          "UsdSchemaRegistry: UsdGenOperator is abstract");
    Check(UsdSchemaRegistry::IsAbstract(TfToken("UsdGenMap")),
          "UsdSchemaRegistry: UsdGenMap is abstract");

    // Fixture: open and verify a non-empty computed extent on the codeless
    // UsdGenDescription (proves the usdGenSchema extent library dlopens).
    UsdStageRefPtr stage = UsdStage::Open(fixture);
    Check(stage != nullptr, "fixture stage opened");
    if (stage) {
        UsdPrim desc = stage->GetPrimAtPath(SdfPath("/Groom/Description"));
        Check(desc.IsValid(), "fixture /Groom/Description exists");
        if (desc.IsValid()) {
            VtVec3fArray extent;
            const bool computed =
                UsdGeomBoundable(desc).ComputeExtent(UsdTimeCode(), &extent);
            Check(computed, "ComputeExtent returned true");
            Check(extent.size() == 2, "extent returned two corners");
            if (extent.size() == 2) {
                const GfVec3f lo = extent[0], hi = extent[1];
                const bool empty =
                    (lo == GfVec3f(0, 0, 0)) && (hi == GfVec3f(0, 0, 0));
                Check(!empty, "extent is non-empty");
                std::printf(
                    "  extent = [(%.3f %.3f %.3f) .. (%.3f %.3f %.3f)]\n",
                    static_cast<double>(lo[0]), static_cast<double>(lo[1]),
                    static_cast<double>(lo[2]),
                    static_cast<double>(hi[0]), static_cast<double>(hi[1]),
                    static_cast<double>(hi[2]));
            }
        }
    }

    std::printf("\ntestUsdGenPluginDiscovery: %s (%d failure%s)\n",
                g_failures ? "FAIL" : "PASS", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
