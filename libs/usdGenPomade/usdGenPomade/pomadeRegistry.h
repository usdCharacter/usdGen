// usdGenPomade — the process-global model registry (plan/18 §2.1).
//
// Before V0 there were two unrelated models: the one Pomade_Create allocated
// inside the DLL for the tools, and the one UsdGenPomadeSceneIndex allocated
// for itself. Nothing joined them, so every edit the tools made was invisible
// in the viewport (plan/18 §0 F1). The registry is that join:
//
//   * Pomade_Create registers its model and gets a model id back;
//     Pomade_Destroy unregisters it.
//   * every scene index attaches on construction and detaches on
//     destruction, as a PomadePublishTarget;
//   * Pomade_Activate names the one model the viewport draws;
//   * Pomade_Publish (main thread) hands the active model to every attached
//     target, which is the ONLY way the viewport learns about a change.
//
// The registry deliberately knows nothing about Hydra: the publish target is
// an interface so this header stays free of pxr includes and the C ABI can
// include it without pulling in the imaging stack.
#ifndef USDGEN_POMADE_REGISTRY_H
#define USDGEN_POMADE_REGISTRY_H

#include "usdGenPomade/api.h"

#include <atomic>
#include <cstdint>

namespace usdGenPomade {

class PomadeModel;
struct PomadeRegistryTable;

// Anything that draws a model. UsdGenPomadeSceneIndex is the only
// implementation; the T1 tests use it directly.
class USDGENPOMADE_API PomadePublishTarget {
public:
    virtual ~PomadePublishTarget();
    // Publish `model` (null = no active model, publish the empty/test
    // state). `dirtyMask` is a PomadeDirty OR that the caller wants
    // republished on top of whatever the model itself reports pending.
    // Returns true when anything was published.
    virtual bool PublishModel(PomadeModel *model, uint32_t dirtyMask) = 0;
    // Census of what is on screen for `level`; false when that level is not
    // published. This is how a tool asks what the viewport actually got,
    // rather than assuming its own edit arrived.
    virtual bool QueryPublishedLevel(int level, int *outFaceCount,
                                     int *outPointCount,
                                     int *outTubeCount) const = 0;
};

class USDGENPOMADE_API PomadeRegistry {
public:
    // Constant-initialised and trivially destructible: Hydra scene indices
    // can be destroyed during static destruction, and DetachIndex must not
    // touch a registry that has already run its own destructor.
    //
    // The tables are copy-on-write, published by atomic pointer: readers
    // pin an epoch, take a reference and never wait; writers build the next
    // table and CAS it in, retiring the old one through the epochs.
    static PomadeRegistry &Get();

    // Model lifetime. Register returns a model id > 0; Unregister on the
    // active model deactivates it first. Registering the same model twice
    // returns the existing id.
    int Register(PomadeModel *model);
    void Unregister(int modelId);
    int ModelCount() const;

    // The one model the viewport draws. `modelId` 0 clears it. Returns
    // false for an unknown id (the active model is left alone).
    bool SetActive(int modelId);
    int GetActiveId() const;
    PomadeModel *Active() const;

    // Scene-index attachment. Attaching twice is a no-op.
    void AttachIndex(PomadePublishTarget *target);
    void DetachIndex(PomadePublishTarget *target);
    int IndexCount() const;

    // Main thread only. Publishes the active model to every attached
    // target and returns how many of them published something.
    int Publish(uint32_t dirtyMask);
    // Ask the attached targets what they published for `level`; the first
    // one that answers wins. False when no target publishes that level.
    bool QueryPublishedLevel(int level, int *outFaceCount, int *outPointCount,
                             int *outTubeCount) const;

private:
    constexpr PomadeRegistry() noexcept = default;
    ~PomadeRegistry() = default;
    PomadeRegistry(PomadeRegistry const &) = delete;
    PomadeRegistry &operator=(PomadeRegistry const &) = delete;

    static PomadeRegistry _instance;
    std::atomic<PomadeRegistryTable const *> _table{nullptr};
    std::atomic<int> _nextId{1};
};

} // namespace usdGenPomade

#endif // USDGEN_POMADE_REGISTRY_H
