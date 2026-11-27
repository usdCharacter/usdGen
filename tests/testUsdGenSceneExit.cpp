// Process-exit lifetime regression.  The holder is constructed before the
// scene service singleton and takes the index after Synchronize, then drops
// it before return so pipeline Shutdown still runs with TBB workers alive.
#include "utils/pxrInterop.h"
#include "usdGenImaging/groomSceneIndexPlugin.h"

#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/retainedSceneIndex.h"

#include <cstdio>

PXR_NAMESPACE_USING_DIRECTIVE

namespace {

struct ExitHolder {
    HdSceneIndexBaseRefPtr input;
    HdSceneIndexBaseRefPtr index;
};

// Must precede first construction of UsdGenSceneService in main.
ExitHolder g_exitHolder;

} // namespace

int main()
{
    auto input = HdRetainedSceneIndex::New();
    const SdfPath root("/__sceneExitGroom");
    input->AddPrims({{root, TfToken("UsdGenGroom"),
                      HdRetainedContainerDataSource::New()}});

    auto index = UsdGenGroomSceneIndex::New(input, 5801);
    auto *owner = dynamic_cast<UsdGenGroomSceneIndex *>(index.operator->());
    if (!owner) {
        std::fprintf(stderr, "scene exit owner cast failed\n");
        return 2;
    }
    owner->Synchronize();
    if (index->GetPrim(root).primType != TfToken("UsdGenGroom")) {
        std::fprintf(stderr, "scene exit groom was not populated\n");
        return 3;
    }

    // Hold the index past the last local, then drop it while the TBB worker
    // market is still alive. A static destructor after TBB's governor has
    // stopped workers waits forever in pipeline Shutdown (the Linux twin of
    // the Windows RtlDllShutdownInProgress hang).
    g_exitHolder.input = input;
    g_exitHolder.index = index;
    std::printf("testUsdGenSceneExit: PASS (retained index through exit)\n");
    std::fflush(stdout);
    g_exitHolder.index.Reset();
    g_exitHolder.input.Reset();
    return 0;
}
