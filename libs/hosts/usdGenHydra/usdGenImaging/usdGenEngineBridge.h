// usdGen imaging — scheduled commit-thread gate + stats-ring drain.
//
// E-2 capture gate: a serial owner records the first Hydra authoring caller
// and reports later off-thread capture requests. Actual engine execution
// belongs to the session's scheduled owner/work nodes, not this caller thread.
// DrainStats copies the engine's immutable stats value into that owner, which
// forwards unseen ring entries to an optional hook exactly once per entry.
//
// Plan: plan/06-imaging.md §3.1 (per-session member of UsdGenImagingSession).
#ifndef USDGEN_IMAGING_ENGINE_BRIDGE_H
#define USDGEN_IMAGING_ENGINE_BRIDGE_H

#include "usdGen/stats.h"

#include "pxr/usd/sdf/path.h"
#include "pxr/pxr.h"

#include <atomic>
#include <functional>
#include <memory>
#include <thread>

PXR_NAMESPACE_OPEN_SCOPE

namespace usdGenImaging {

class UsdGenEngineBridge
{
public:
    using TimingHook = std::function<void(usdGen::UsdGenCommitTiming const &)>;

    UsdGenEngineBridge();
    ~UsdGenEngineBridge();
    UsdGenEngineBridge(UsdGenEngineBridge const&) = delete;
    UsdGenEngineBridge& operator=(UsdGenEngineBridge const&) = delete;

    /// Enqueues recording of the thread allowed to run engine commits (E-2).
    void RegisterCommitThread(std::thread::id tid);

    /// False until the first commit registers a thread.
    bool HasCommitThread() const;
    bool IsCommitThread() const;

    /// Enqueues validation before every engine Commit; owner reports violations.
    void GateCommit(SdfPath const &descriptionPath, char const *where);

    /// Enqueues installation of the timing sink; empty removes it.
    void SetTimingHook(TimingHook hook);

    /// Enqueues a copied stats value; forwards unseen entries, oldest first.
    void DrainStats(usdGen::UsdGenStats const &stats);

    /// External test/shutdown boundary only. Never call from an owner callback.
    void Drain();

    // Opaque owner state.  Kept public only so the implementation's
    // process-level owner can retain it in queued commands.
    struct State;

private:
    std::shared_ptr<State> _state;
};

}  // namespace usdGenImaging

PXR_NAMESPACE_CLOSE_SCOPE

#endif  // USDGEN_IMAGING_ENGINE_BRIDGE_H
