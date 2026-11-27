#include "usdGenImaging/usdGenEngineBridge.h"
#include "usdGen/api/sharedCell.h"

#include "usdGen/api/pipeline.h"
#include "pxr/base/tf/diagnostic.h"

#include <memory>
#include <string>

PXR_NAMESPACE_OPEN_SCOPE

namespace usdGenImaging {


namespace {

struct BridgeOwner {
    usdGen::UsdGenExecutionRuntime runtime{8};
    usdGen::UsdGenExecutionPipeline pipeline{runtime};
};

std::atomic<BridgeOwner *> g_bridgeOwner{nullptr};

BridgeOwner &_BridgeOwner()
{
    // The process owner, rather than an embedded pipeline, owns queued
    // commands.  A bridge may therefore be destroyed from a publication
    // callback: its State is retained by the command and no destructor waits.
    //
    // Leaked on purpose, like every process-lifetime arena owner in this
    // library (see ImagingRuntime in usdGenImagingSession.cpp): a static
    // destructor here runs at DLL_PROCESS_DETACH on Windows, after ExitProcess
    // has killed the TBB workers, so its drain could never complete. Callers
    // that need queued commands delivered before exit call Drain explicitly.
    return usdGen::UsdGenLazyLeaked(g_bridgeOwner);
}

} // namespace

struct UsdGenEngineBridge::State {
    struct Snapshot {
        std::thread::id commitThread;
        bool hasCommitThread = false;
        TimingHook timingHook;
        size_t seenEntries = 0;
        bool firstDrain = true;
    };
    usdGen::UsdGenSharedCell<Snapshot const> published{std::make_shared<Snapshot const>()};
};

namespace {

using Snapshot = UsdGenEngineBridge::State::Snapshot;

std::shared_ptr<Snapshot const> _Snapshot(
    std::shared_ptr<UsdGenEngineBridge::State> const &state)
{
    return state->published.Load();
}

void _Publish(std::shared_ptr<UsdGenEngineBridge::State> const &state,
              std::shared_ptr<Snapshot> next)
{
    state->published.Store(std::move(next));
}

void _Post(std::function<void()> command)
{
    // PostCommand is intentionally nonblocking even when invoked by an owner
    // callback.  All bridge mutation and hook delivery remains ordered.
    if (!_BridgeOwner().pipeline.PostCommand(std::move(command)))
        TF_CODING_ERROR("usdGen: imaging bridge owner rejected a command");
}

} // namespace

UsdGenEngineBridge::UsdGenEngineBridge() : _state(std::make_shared<State>()) {}
UsdGenEngineBridge::~UsdGenEngineBridge() = default;

void UsdGenEngineBridge::RegisterCommitThread(std::thread::id tid)
{
    auto state = _state;
    _Post([state, tid] {
        auto current = _Snapshot(state);
        if (current->hasCommitThread) return;
        auto next = std::make_shared<Snapshot>(*current);
        next->commitThread = tid;
        next->hasCommitThread = true;
        _Publish(state, std::move(next));
    });
}

bool UsdGenEngineBridge::HasCommitThread() const
{
    return _Snapshot(_state)->hasCommitThread;
}

bool UsdGenEngineBridge::IsCommitThread() const
{
    auto snapshot = _Snapshot(_state);
    return !snapshot->hasCommitThread ||
        std::this_thread::get_id() == snapshot->commitThread;
}

void UsdGenEngineBridge::GateCommit(SdfPath const &descriptionPath, char const *where)
{
    auto state = _state;
    auto tid = std::this_thread::get_id();
    std::string location = where ? where : "unknown";
    _Post([state, tid, descriptionPath, location=std::move(location)] {
        auto current = _Snapshot(state);
        if (!current->hasCommitThread) {
            auto next = std::make_shared<Snapshot>(*current);
            next->commitThread = tid;
            next->hasCommitThread = true;
            _Publish(state, std::move(next));
        } else if (tid != current->commitThread) {
            TF_CODING_ERROR(
                "usdGen: engine commit for '%s' from a thread other than the registered "
                "commit thread (%s); Hydra commit must stay on the scene-authoring thread "
                "(E-2).", descriptionPath.GetText(), location.c_str());
        }
    });
}

void UsdGenEngineBridge::SetTimingHook(TimingHook hook)
{
    auto state = _state;
    _Post([state, hook=std::move(hook)]() mutable {
        auto next = std::make_shared<Snapshot>(*_Snapshot(state));
        next->timingHook = std::move(hook);
        _Publish(state, std::move(next));
    });
}

void UsdGenEngineBridge::DrainStats(usdGen::UsdGenStats const &stats)
{
    // Stats is an owner-published value at its source.  Copy it before the
    // command is queued so the bridge never borrows a caller's mutable ring.
    auto state = _state;
    auto snapshot = std::make_shared<usdGen::UsdGenStats>(stats);
    _Post([state, snapshot] {
        auto current = _Snapshot(state);
        const size_t n = snapshot->ring.size();
        const size_t head = n ? snapshot->ringHead % n : 0;
        const size_t first = n && !current->firstDrain ?
            current->seenEntries % n : head;
        TimingHook hook = current->timingHook;
        auto next = std::make_shared<Snapshot>(*current);
        next->seenEntries = head;
        next->firstDrain = false;
        _Publish(state, std::move(next));
        if (!hook || !n) return;
        for (size_t i = first; i != head; i = (i + 1) % n) hook(snapshot->ring[i]);
    });
}

void UsdGenEngineBridge::Drain()
{
    _BridgeOwner().pipeline.Drain();
}

}  // namespace usdGenImaging

PXR_NAMESPACE_CLOSE_SCOPE
