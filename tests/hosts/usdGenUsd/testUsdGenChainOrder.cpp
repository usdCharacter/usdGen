// testUsdGenChainOrder — M0 vertical-demo boundary test (T1).
//
// Proves where UsdGenGroomSceneIndex lands in a real Hydra render chain:
//   * strictly AFTER HdsiSceneGlobalsSceneIndex (so it sees currentFrame in
//     usdview *and* usdrecord),
//   * strictly BEFORE every observed Hdsi*/HdSt*/Storm renderer node
//     (Hybrid policy — the default),
//   * strictly BEFORE the synthetic hdPrman:motionBlur fixture,
//   * strictly AFTER the whole UsdImaging/UsdSkel chain.
//
// Under JsonMetadataOnly the hdsi ordering is governed by their own JSON
// metadata, so we assert only what usdGen's plugInfo promises:
//   - usdGen after HdsiSceneGlobalsSceneIndex
//   - usdGen before the synthetic hdPrman:motionBlur node
//   - all UsdImaging/UsdSkel nodes upstream of usdGen
//
// All boundary nodes are asserted unconditionally — absence is a failure
// (sol S-3 / X-2: the old conditional checks passed vacuously).
//
// The chain mirrors the usdImagingGL engine structure: the usdImaging
// terminal is wrapped in HdsiSceneGlobalsSceneIndex
// (usdImagingGL/engine.cpp:155), then the registry appends the
// renderer-level plugins (engine.cpp:1778, hd/renderIndex.cpp:208).
// Headless: no render delegate or GL context is required.
//
// Plugin discovery: usdGen and the fake hdPrman fixture are NOT linked into
// this executable. They are discovered through PXR_PLUGINPATH_NAME (usdGen
// build-tree resources, OpenUSD plugin/usd + lib/usd, fixture resources)
// and dlopen'd by the plugin registry while the test runs, so the
// TfErrorMark below catches any Tf coding error during that load — e.g.
// the double type registration sol S-4 found.

#include "pxr/pxr.h"
#include "pxr/base/arch/demangle.h"
#include "pxr/base/tf/errorMark.h"
#include "pxr/imaging/hd/filteringSceneIndex.h"
#include "pxr/imaging/hd/sceneIndex.h"
#include "pxr/imaging/hd/sceneIndexPluginRegistry.h"
#include "pxr/imaging/hdsi/sceneGlobalsSceneIndex.h"
#include "pxr/usd/usd/stage.h"
#include "pxr/usdImaging/usdImaging/sceneIndices.h"

#include <cstdio>
#include <string>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

int g_failures = 0;
void Check(bool ok, const char *what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what); }
    else     { std::printf("ok:   %s\n", what); }
}

struct ChainNode
{
    std::string type;     // demangled class name, namespace stripped
    std::string display;  // GetDisplayName()
};

std::string ShortName(const HdSceneIndexBaseRefPtr &si)
{
    if (!si) return "<null>";
    std::string t = ArchGetDemangled(typeid(*si).name());
    const auto p = t.rfind("::");
    if (p != std::string::npos) t = t.substr(p + 1);
    return t;
}

// Walk the chain from the terminal (index 0, closest to the renderer)
// toward the stage (deepest index), descending actual input edges through
// both filtering and merging (HdMergingSceneIndex derives from
// HdFilteringSceneIndexBase in this API) nodes.
std::vector<ChainNode>
WalkChain(const HdSceneIndexBaseRefPtr &terminal)
{
    std::vector<ChainNode> nodes;
    std::vector<HdSceneIndexBaseRefPtr> frontier;
    if (terminal) frontier.push_back(terminal);
    for (size_t i = 0; i < frontier.size(); ++i) {
        const HdSceneIndexBaseRefPtr si = frontier[i];
        ChainNode node;
        node.type = ShortName(si);
        node.display = si->GetDisplayName();
        nodes.push_back(node);
        const HdSceneIndexBase *raw = si.operator->();
        if (const HdFilteringSceneIndexBase *filter =
                dynamic_cast<const HdFilteringSceneIndexBase *>(raw)) {
            for (const auto &in : filter->GetInputScenes()) {
                if (in) frontier.push_back(in);
            }
        }
    }
    return nodes;
}

int FindNode(const std::vector<ChainNode> &nodes, const std::string &substr)
{
    for (size_t i = 0; i < nodes.size(); ++i)
        if (nodes[i].type.find(substr) != std::string::npos ||
            nodes[i].display.find(substr) != std::string::npos)
            return static_cast<int>(i);
    return -1;
}

// Usd-side nodes: the UsdImaging/UsdSkel chain proper, plus the two
// Hd*/Hdsi* types that usdImaging builds internally (NOT renderer plugins).
bool IsUsdSide(const ChainNode &n)
{
    return n.type.find("UsdImaging") != std::string::npos ||
           n.type.find("UsdSkel") != std::string::npos ||
           n.type.find("HdNoticeBatching") != std::string::npos ||
           n.type.find("HdsiLocatorCaching") != std::string::npos;
}

bool IsSceneGlobals(const ChainNode &n)
{
    return n.type.find("HdsiSceneGlobals") != std::string::npos;
}

// Mirrors usdImagingGL/engine.cpp: sceneGlobals wraps the usdImaging
// terminal, then the renderer-level registry plugins append on top.
HdSceneIndexBaseRefPtr
BuildChain(const HdSceneIndexBaseRefPtr &usdTerminal)
{
    HdSceneIndexBaseRefPtr withGlobals =
        HdsiSceneGlobalsSceneIndex::New(usdTerminal);
    return HdSceneIndexPluginRegistry::GetInstance()
        .AppendSceneIndicesForRenderer("GL", withGlobals);
}

void
PrintChain(const char *label, const std::vector<ChainNode> &nodes)
{
    std::printf("chain (%s, terminal -> upstream), %zu nodes:\n",
                label, nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i)
        std::printf("  %2zu. %s%s\n", i, nodes[i].type.c_str(),
                    nodes[i].display.empty() ? "" :
                        (std::string(" [") + nodes[i].display + "]").c_str());
    std::printf("\n");
}

// Core presence + boundary checks shared by both policy cases.
// Walk index 0 = render terminal; larger indices = closer to stage.
// "After X" in dataflow = smaller index than X (closer to terminal).
// "Before X" in dataflow = larger index than X (closer to stage).
void
CheckCore(const std::vector<ChainNode> &nodes, int ours, int sg, int fake,
          int usdStage)
{
    // usdGen strictly after sceneGlobals (sees currentFrame)
    Check(ours < sg,
          "usdGen sits strictly after HdsiSceneGlobalsSceneIndex "
          "(sees currentFrame in usdview and usdrecord)");

    // usdGen strictly before the synthetic hdPrman:motionBlur node
    Check(fake > 0 && fake < ours,
          "usdGen sits strictly before the synthetic hdPrman:motionBlur node "
          "(ordering.before honored by the policy)");

    // All UsdImaging/UsdSkel nodes are upstream of usdGen (dataflow: before usdGen)
    int usdSideOnRendererSide = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const int idx = static_cast<int>(i);
        if (idx == ours || IsSceneGlobals(nodes[i])) continue;
        if (IsUsdSide(nodes[i]) && idx < ours) {
            ++usdSideOnRendererSide;
            std::printf("  (usd-side node on renderer side of usdGen: %s)\n",
                        nodes[i].type.c_str());
        }
    }
    Check(usdSideOnRendererSide == 0,
          "every UsdImaging/UsdSkel node is strictly upstream of usdGen");

    // usdStage is the deepest node
    Check(usdStage >= ours && usdStage == static_cast<int>(nodes.size()) - 1,
          "UsdImagingStageSceneIndex is the deepest node (upstream of usdGen)");
}

// Full boundary check: every Hdsi*/HdSt*/Storm renderer node is strictly
// downstream (terminal-side) of usdGen. Valid under Hybrid (default) policy.
void
CheckAllRendererNodes(const std::vector<ChainNode> &nodes, int ours)
{
    int rendererOnUsdSide = 0;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const int idx = static_cast<int>(i);
        if (idx == ours || IsSceneGlobals(nodes[i])) continue;
        if (!IsUsdSide(nodes[i]) && idx > ours) {
            ++rendererOnUsdSide;
            std::printf("  (renderer node on usd side of usdGen: %s)\n",
                        nodes[i].type.c_str());
        }
    }
    Check(rendererOnUsdSide == 0,
          "every observed Hdsi*/HdSt*/Storm renderer node is strictly "
          "downstream of usdGen");
}

}  // namespace

int main()
{
    std::printf("=== testUsdGenChainOrder ===\n");
    std::printf("PXR_PLUGINPATH_NAME=%s\n\n",
                std::getenv("PXR_PLUGINPATH_NAME")
                    ? std::getenv("PXR_PLUGINPATH_NAME") : "<unset>");

    // Any Tf coding error from here on (including plugin-library load inside
    // the registry queries below) fails the test (sol S-4 / X-8 regression
    // guard against double type registration).
    TfErrorMark errorMark;

    UsdStageRefPtr stage = UsdStage::CreateInMemory();
    UsdImagingCreateSceneIndicesInfo info;
    info.stage = stage;
    const UsdImagingSceneIndices sis = UsdImagingCreateSceneIndices(info);
    const HdSceneIndexBaseRefPtr usdTerminal = sis.finalSceneIndex;
    Check(usdTerminal != nullptr, "UsdImaging chain built");
    if (!usdTerminal) return 1;

    // ---- Case 1: default (Hybrid) ordering policy ----
    {
        const auto nodes = WalkChain(BuildChain(usdTerminal));
        PrintChain("default (Hybrid) policy", nodes);

        const int ours     = FindNode(nodes, "UsdGenGroomSceneIndex");
        const int pomade    = FindNode(nodes, "UsdGenPomadeSceneIndex");
        const int sg       = FindNode(nodes, "HdsiSceneGlobalsSceneIndex");
        const int fake     = FindNode(nodes, "FakeHdPrmanMotionBlurSceneIndex");
        const int usdStage = FindNode(nodes, "UsdImagingStageSceneIndex");
        const int fwd      = FindNode(nodes, "HdDependencyForwardingSceneIndex");
        const int implicit = FindNode(nodes, "HdsiImplicitSurfaceSceneIndex");
        const int matbind  = FindNode(nodes, "HdsiMaterialBindingResolvingSceneIndex");
        int storm          = FindNode(nodes, "HdSt");
        if (storm < 0) storm = FindNode(nodes, "Storm");

        // Mandatory presence — no "if found" guards.
        Check(ours > 0, "UsdGenGroomSceneIndex present in renderer chain");
        Check(sg > 0, "HdsiSceneGlobalsSceneIndex present in renderer chain");
        Check(usdStage >= 0, "UsdImagingStageSceneIndex present");
        Check(fwd >= 0, "HdDependencyForwardingSceneIndex present (render terminal)");
        Check(implicit > 0, "HdsiImplicitSurfaceSceneIndex present (hdsi renderer stack)");
        Check(matbind > 0, "HdsiMaterialBindingResolvingSceneIndex present (hdsi renderer stack)");
        Check(storm >= 0, "an HdSt*/Storm renderer node present (renderer stack loaded)");
        Check(fake > 0, "FakeHdPrmanMotionBlurSceneIndex present (synthetic hdPrman fixture)");
        Check(pomade > 0, "UsdGenPomadeSceneIndex present in renderer chain (plan/17 P0)");

        if (ours > 0) {
            CheckCore(nodes, ours, sg, fake, usdStage);
            CheckAllRendererNodes(nodes, ours);
            // Pomade phase 0, after the groom index (plugInfo ordering): in
            // dataflow the pomade index is downstream of the groom index, i.e.
            // terminal-side in walk order.
            Check(pomade > 0 && pomade < ours,
                  "UsdGenPomadeSceneIndex sits strictly after (downstream of) "
                  "UsdGenGroomSceneIndex");
        }
    }

    // ---- Case 2: JSON-metadata-only ordering policy ----
    {
        HdSceneIndexPluginRegistry::GetInstance().SetPluginOrderingPolicy(
            HdSceneIndexPluginRegistry::PluginOrderingPolicy::JsonMetadataOnly);

        const auto nodes = WalkChain(BuildChain(usdTerminal));
        PrintChain("JsonMetadataOnly policy", nodes);

        const int ours     = FindNode(nodes, "UsdGenGroomSceneIndex");
        const int pomade    = FindNode(nodes, "UsdGenPomadeSceneIndex");
        const int sg       = FindNode(nodes, "HdsiSceneGlobalsSceneIndex");
        const int fake     = FindNode(nodes, "FakeHdPrmanMotionBlurSceneIndex");
        const int usdStage = FindNode(nodes, "UsdImagingStageSceneIndex");

        Check(ours > 0, "usdGen still present under JsonMetadataOnly");
        Check(pomade > 0, "pomade still present under JsonMetadataOnly");
        Check(sg > 0, "HdsiSceneGlobalsSceneIndex still present under JsonMetadataOnly");

        if (ours > 0 && sg > 0) {
            CheckCore(nodes, ours, sg, fake, usdStage);
            if (pomade > 0) {
                Check(pomade < ours,
                      "pomade still downstream of the groom index under "
                      "JsonMetadataOnly (tag ordering)");
            }
            // Under JsonMetadataOnly, hdsi node positions are governed by
            // their own JSON metadata, not by usdGen's C++ registration
            // order. The usdGen contract only promises: after sceneGlobals,
            // before hdPrman:motionBlur, after the UsdImaging chain.
        }
    }

    Check(errorMark.IsClean(),
          "no Tf coding errors (double type registration, bad types, ...)");

    std::printf("\ntestUsdGenChainOrder: %s (%d failure%s)\n",
                g_failures ? "FAIL" : "PASS", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
