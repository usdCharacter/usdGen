// usdGenPomade — the private context layouts behind the C ABI handles.
//
// The one definition shared by every ABI translation unit (pomadeApi.cpp,
// pomadeApiStage.cpp). Each opaque handle is exactly one of these.
#ifndef USDGEN_POMADE_API_IMPL_H
#define USDGEN_POMADE_API_IMPL_H

#include "usdGenPomade/pomadeBake.h"
#include "usdGenPomade/pomadeCommit.h"
#include "usdGenPomade/pomadeModel.h"
#include "usdGenPomade/pomadeRegistry.h"
#include "usdGenPomade/pomadeTransport.h"

#include <memory>
#include <string>

namespace usdGenPomade {

struct PomadeModelContextImpl {
    PomadeModel model;
    PomadePinnedStaging positions;
    PomadePinnedStaging normals;
    // The registry id this model was created under (plan/18 §2.1). The
    // scene indices find the model through it; without it a tool's edits
    // never reach the viewport.
    int modelId = 0;

    PomadeModelContextImpl() { modelId = PomadeRegistry::Get().Register(&model); }
    ~PomadeModelContextImpl() { PomadeRegistry::Get().Unregister(modelId); }
    PomadeModelContextImpl(PomadeModelContextImpl const &) = delete;
    PomadeModelContextImpl &operator=(PomadeModelContextImpl const &) = delete;
};

struct PomadeCommitterContextImpl {
    std::unique_ptr<PomadeCommitter> committer;
};

struct PomadeBakeContextImpl {
    PomadeModel *model = nullptr;  // non-owning (outlives us)
    std::unique_ptr<PomadeBakeWorker> worker;
    std::string outDir;
    std::string baseName = "regionMap";
    int resOverride = -1;
    int levelCount = 1;
};

}  // namespace usdGenPomade

#endif  // USDGEN_POMADE_API_IMPL_H
