// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGenPomade — the published state of a PomadeModel: an immutable,
// refcounted copy of everything a commit serialises. The authoring thread
// captures it (PomadeModel::PublishState / CaptureState); any other thread
// reads it (AcquireState) and never touches the live model.
//
// Copy-on-write: a state is never changed after capture; the next edit's
// state is a new object, and the one it displaces is retired through the
// usdGen epochs. Internal to usdGenPomade's sources (it includes usdGen).
#ifndef USDGEN_POMADE_MODEL_STATE_H
#define USDGEN_POMADE_MODEL_STATE_H

#include "usdGenPomade/pomadeModel.h"

#include "usdGen/epoch.h"

#include <map>
#include <memory>
#include <vector>

namespace usdGenPomade {

struct PomadeModelState : usdGen::UsdGenRefCounted {
    uint64_t version = 0;
    PomadeModel::TubeSnapshot tube;  // the first L1 root
    bool generatedCurvesSuppressed = false;
    PomadeModel::OutputSettings output;
    int tubeRegionId = -1;
    std::vector<int> l1TubeIds;
    std::vector<int> tubeIds;
    // Every tube but 0 that GetTubeRecord answers for.
    std::map<int, PomadeModel::TubeRecord> records;
    PomadeModel::GraphSnapshot graph;
    std::shared_ptr<PomadeScalpMesh const> scalp;
};

using PomadeModelStateRef = usdGen::UsdGenRef<PomadeModelState const>;

inline PomadeModelStateRef
PomadeAcquireModelState(PomadeModel const &model)
{
    return PomadeModelStateRef::Adopt(model.AcquireState());
}

inline PomadeModelStateRef
PomadeCaptureModelState(PomadeModel const &model)
{
    return PomadeModelStateRef::Adopt(model.CaptureState());
}

}  // namespace usdGenPomade

#endif  // USDGEN_POMADE_MODEL_STATE_H
