#include "usdGen/executionRetirement.h"
#include "testBackend.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {
struct Counts { std::atomic<int> complete{0}, destroy{0}, quarantine{0}, destruct{0}; };
struct Payload final : UsdGenExecutionRetirementPayload {
    Payload(Counts& counts, bool complete = true, bool destroy = true)
        : counts(counts), completeResult(complete), destroyResult(destroy) {}
    ~Payload() override { ++counts.destruct; }
    bool IsComplete() noexcept override { ++counts.complete; return completeResult; }
    bool Destroy() noexcept override { ++counts.destroy; return destroyResult; }
    void Quarantine() noexcept override { ++counts.quarantine; }
    Counts& counts;
    bool completeResult;
    bool destroyResult;
};

struct ThreadPayload final : UsdGenExecutionRetirementPayload {
    ThreadPayload(std::atomic<bool>& onCaller, std::atomic<bool>& finished,
                  std::thread::id caller)
        : onCaller(onCaller), finished(finished), caller(caller) {}
    bool IsComplete() noexcept override { return true; }
    bool Destroy() noexcept override {
        onCaller.store(std::this_thread::get_id() == caller, std::memory_order_release);
        finished.store(true, std::memory_order_release);
        return true;
    }
    void Quarantine() noexcept override {}
    std::atomic<bool>& onCaller;
    std::atomic<bool>& finished;
    std::thread::id caller;
};

struct HeldDestroyPayload final : UsdGenExecutionRetirementPayload {
    HeldDestroyPayload(Counts& counts, std::atomic<bool>& entered,
                       std::atomic<bool>& release, std::atomic<bool>& timedOut)
        : counts(counts), entered(entered), release(release), timedOut(timedOut) {}
    ~HeldDestroyPayload() override { ++counts.destruct; }
    bool IsComplete() noexcept override { ++counts.complete; return true; }
    bool Destroy() noexcept override {
        ++counts.destroy;
        entered.store(true, std::memory_order_release);
        auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!release.load(std::memory_order_acquire)) {
            if (std::chrono::steady_clock::now() >= deadline) {
                timedOut.store(true, std::memory_order_release);
                break;
            }
            std::this_thread::yield();
        }
        return true;
    }
    void Quarantine() noexcept override { ++counts.quarantine; }
    Counts& counts;
    std::atomic<bool>& entered;
    std::atomic<bool>& release;
    std::atomic<bool>& timedOut;
};

// The parent is destroyed on the retirement worker. Its Destroy method creates
// a child retirement, proving Drain accounts for work admitted during cleanup.
struct NestedPayload final : UsdGenExecutionRetirementPayload {
    NestedPayload(std::shared_ptr<UsdGenExecutionRetirementService> service,
                  Counts& parent, Counts& child)
        : service(std::move(service)), parent(parent), child(child) {}
    bool IsComplete() noexcept override { return true; }
    bool Destroy() noexcept override {
        ++parent.destroy;
        auto ticket = service->TryReserve();
        if (!ticket) return false;
        auto signal = ticket->MakeSignal();
        if (!ticket->Retire(std::make_unique<Payload>(child))) return false;
        signal.SignalSuccess();
        return true;
    }
    void Quarantine() noexcept override { ++parent.quarantine; }
    std::shared_ptr<UsdGenExecutionRetirementService> service;
    Counts& parent;
    Counts& child;
};
std::shared_ptr<UsdGenExecutionRetirementService> Service(int device, size_t capacity) {
    return GetOrCreateUsdGenExecutionRetirementService(
        {usdGenTestUtils::kTestBackendA, device}, {capacity});
}
}

int main() {
    // Early completion, duplicate completion, and capacity recovery after a
    // successful worker-side destroy.
    auto service = Service(719320, 1);
    CHECK(service && !GetOrCreateUsdGenExecutionRetirementService(
        {usdGenTestUtils::kTestBackendA, 719320}, {2}));
    Counts early;
    auto ticket = service->TryReserve(); CHECK(ticket && !service->TryReserve());
    auto signal = ticket->MakeSignal();
    signal.SignalSuccess(); signal.SignalFailure(); signal.SignalSuccess();
    CHECK(ticket->Retire(std::make_unique<Payload>(early)));
    service->Drain();
    CHECK(early.complete == 1 && early.destroy == 1 && early.destruct == 1 && early.quarantine == 0);
    CHECK(service->TryReserve());

    // Retire-before-signal stays admitted; Drain waits through the flow graph
    // rather than polling. The signalling thread models a backend callback.
    auto delayed = Service(719321, 1); Counts delayedCounts;
    auto delayedTicket = delayed->TryReserve(); CHECK(delayedTicket);
    auto delayedSignal = delayedTicket->MakeSignal();
    CHECK(delayedTicket->Retire(std::make_unique<Payload>(delayedCounts)));
    std::thread later([delayedSignal] {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        delayedSignal.SignalSuccess();
    });
    delayed->Drain(); later.join();
    CHECK(delayedCounts.complete == 1 && delayedCounts.destroy == 1 && delayedCounts.destruct == 1);

    // A failure is first-wins and permanently consumes the bounded slot.
    auto failed = Service(719322, 1); Counts failedCounts;
    auto failedTicket = failed->TryReserve(); CHECK(failedTicket);
    auto failedSignal = failedTicket->MakeSignal();
    CHECK(failedTicket->Retire(std::make_unique<Payload>(failedCounts)));
    failedSignal.SignalFailure(); failedSignal.SignalSuccess();
    failed->Drain();
    CHECK(failedCounts.complete == 0 && failedCounts.destroy == 0 &&
          failedCounts.destruct == 0 && failedCounts.quarantine == 1);
    CHECK(!failed->TryReserve());

    // A signal copied from the prior slot generation cannot complete a newly
    // admitted payload after the first payload has freed that slot.
    auto stale = Service(719323, 1); Counts one, two;
    auto first = stale->TryReserve(); CHECK(first);
    auto staleSignal = first->MakeSignal();
    CHECK(first->Retire(std::make_unique<Payload>(one))); staleSignal.SignalSuccess(); stale->Drain();
    auto second = stale->TryReserve(); CHECK(second);
    auto secondSignal = second->MakeSignal();
    CHECK(second->Retire(std::make_unique<Payload>(two)));
    staleSignal.SignalSuccess(); // stale generation: must be ignored
    secondSignal.SignalFailure(); stale->Drain();
    CHECK(two.complete == 0 && two.destroy == 0 && two.quarantine == 1);

    // Releasing an unretired ticket advances its generation. Its callback is
    // stale and must not poison the newly free/reused slot.
    auto abandoned = Service(719326, 1); Counts abandonedCounts;
    auto abandonedTicket = abandoned->TryReserve(); CHECK(abandonedTicket);
    auto abandonedSignal = abandonedTicket->MakeSignal();
    abandonedTicket.reset();
    auto recovered = abandoned->TryReserve(); CHECK(recovered);
    auto recoveredSignal = recovered->MakeSignal();
    abandonedSignal.SignalSuccess();
    CHECK(recovered->Retire(std::make_unique<Payload>(abandonedCounts)));
    recoveredSignal.SignalSuccess(); abandoned->Drain();
    CHECK(abandonedCounts.destroy == 1 && abandonedCounts.destruct == 1 &&
          abandonedCounts.quarantine == 0);

    // Backend verification and cleanup failures each make one terminal claim.
    auto incomplete = Service(719327, 1); Counts incompleteCounts;
    auto incompleteTicket = incomplete->TryReserve(); CHECK(incompleteTicket);
    auto incompleteSignal = incompleteTicket->MakeSignal();
    CHECK(incompleteTicket->Retire(std::make_unique<Payload>(incompleteCounts, false)));
    incompleteSignal.SignalSuccess(); incomplete->Drain();
    CHECK(incompleteCounts.complete == 1 && incompleteCounts.destroy == 0 &&
          incompleteCounts.destruct == 0 &&
          incompleteCounts.quarantine == 1);
    auto destroyFailure = Service(719328, 1); Counts destroyFailureCounts;
    auto destroyFailureTicket = destroyFailure->TryReserve(); CHECK(destroyFailureTicket);
    auto destroyFailureSignal = destroyFailureTicket->MakeSignal();
    CHECK(destroyFailureTicket->Retire(
        std::make_unique<Payload>(destroyFailureCounts, true, false)));
    destroyFailureSignal.SignalSuccess(); destroyFailure->Drain();
    CHECK(destroyFailureCounts.complete == 1 && destroyFailureCounts.destroy == 1 &&
          destroyFailureCounts.destruct == 0 &&
          destroyFailureCounts.quarantine == 1);

    // Signal and Retire may overlap; installation prevents a signal from
    // observing a null payload or quarantining the valid handoff.
    auto raced = Service(719324, 1); Counts race;
    auto raceTicket = raced->TryReserve(); CHECK(raceTicket);
    auto raceSignal = raceTicket->MakeSignal();
    std::thread concurrent([raceSignal] { raceSignal.SignalSuccess(); });
    CHECK(raceTicket->Retire(std::make_unique<Payload>(race)));
    concurrent.join(); raced->Drain();
    CHECK(race.complete == 1 && race.destroy == 1 && race.quarantine == 0);

    // Repeat the handoff race enough times to exercise slot reuse and verify
    // that successful cleanup returns its sole capacity every iteration.
    auto repeated = Service(719329, 1); Counts repeatedCounts;
    for (int i = 0; i != 1000; ++i) {
        auto repeatedTicket = repeated->TryReserve(); CHECK(repeatedTicket);
        auto repeatedSignal = repeatedTicket->MakeSignal();
        std::thread signalThread([repeatedSignal] { repeatedSignal.SignalSuccess(); });
        CHECK(repeatedTicket->Retire(std::make_unique<Payload>(repeatedCounts)));
        signalThread.join(); repeated->Drain();
    }
    CHECK(repeatedCounts.complete == 1000 && repeatedCounts.destroy == 1000 &&
          repeatedCounts.destruct == 1000 &&
          repeatedCounts.quarantine == 0 && repeated->TryReserve());

    // The caller never performs backend cleanup; the dedicated TBB worker does.
    auto worker = Service(719330, 1); std::atomic<bool> destroyedOnCaller{true}, workerFinished{false};
    auto workerTicket = worker->TryReserve(); CHECK(workerTicket);
    auto workerSignal = workerTicket->MakeSignal();
    CHECK(workerTicket->Retire(
        std::make_unique<ThreadPayload>(destroyedOnCaller, workerFinished,
                                        std::this_thread::get_id())));
    workerSignal.SignalSuccess();
    auto const workerDeadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!workerFinished.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < workerDeadline)
        std::this_thread::yield();
    bool const autonomous = workerFinished.load(std::memory_order_acquire);
    worker->Drain();
    CHECK(autonomous && !destroyedOnCaller.load(std::memory_order_acquire));

    // Child admission during parent cleanup is retained by the same Drain.
    auto nested = Service(719331, 2); Counts parentCounts, childCounts;
    auto parentTicket = nested->TryReserve(); CHECK(parentTicket);
    auto parentSignal = parentTicket->MakeSignal();
    CHECK(parentTicket->Retire(
        std::make_unique<NestedPayload>(nested, parentCounts, childCounts)));
    parentSignal.SignalSuccess(); nested->Drain();
    CHECK(parentCounts.destroy == 1 && parentCounts.quarantine == 0 &&
          childCounts.destroy == 1 && childCounts.quarantine == 0);

    // Shutdown rejects future admission but outstanding pre-admitted tickets
    // retain their payload and can complete safely afterwards.
    auto shutdown = Service(719325, 1); Counts closed;
    auto closeTicket = shutdown->TryReserve(); CHECK(closeTicket);
    auto closeSignal = closeTicket->MakeSignal(); shutdown->Shutdown();
    CHECK(!shutdown->TryReserve());
    CHECK(closeTicket->Retire(std::make_unique<Payload>(closed)));
    closeSignal.SignalSuccess(); shutdown->Drain();
    CHECK(closed.destroy == 1 && closed.quarantine == 0);

    // Backend quiescence is stronger than Shutdown: an unsignalled retired
    // payload is retained without any payload virtual call, and a ticket held
    // across the boundary can still transfer its payload without dispatch.
    auto quiesced = Service(719332, 2); Counts pendingCounts, lateCounts;
    auto pending = quiesced->TryReserve(); CHECK(pending);
    auto pendingSignal = pending->MakeSignal();
    CHECK(pending->Retire(std::make_unique<Payload>(pendingCounts)));
    quiesced->QuiesceBackend();
    CHECK(pendingCounts.complete == 0 && pendingCounts.destroy == 0 &&
          pendingCounts.quarantine == 0 && pendingCounts.destruct == 0);
    pendingSignal.SignalSuccess(); pendingSignal.SignalFailure();
    CHECK(pendingCounts.complete == 0 && pendingCounts.destroy == 0 &&
          pendingCounts.quarantine == 0 && pendingCounts.destruct == 0);
    auto late = quiesced->TryReserve(); CHECK(!late);

    // Reserve before close, then install and signal after it: Retire consumes
    // the old ticket and releases its temporary Drain credit without calling
    // IsComplete, Destroy, or Quarantine.
    auto preclosed = Service(719333, 1); Counts preclosedCounts;
    auto preclosedTicket = preclosed->TryReserve(); CHECK(preclosedTicket);
    auto preclosedSignal = preclosedTicket->MakeSignal();
    preclosed->QuiesceBackend();
    CHECK(preclosedTicket->Retire(std::make_unique<Payload>(preclosedCounts)));
    preclosedSignal.SignalSuccess(); preclosedSignal.SignalFailure();
    preclosed->Drain();
    CHECK(preclosedCounts.complete == 0 && preclosedCounts.destroy == 0 &&
          preclosedCounts.quarantine == 0 && preclosedCounts.destruct == 0);

    // Cleanup which crossed the boundary before quiesce is waited, rather
    // than having its Drain credit stolen by the close scan.
    auto activeClose = Service(719334, 2); Counts activeCounts;
    std::atomic<bool> entered{false}, release{false}, quiesceReturned{false}, heldTimedOut{false};
    struct ReleaseGuard {
        std::atomic<bool>& release;
        std::shared_ptr<UsdGenExecutionRetirementService> service;
        ~ReleaseGuard() {
            // This guard exists before the first CHECK which can leave a
            // held cleanup running. It keeps the service and payload's
            // referenced test state alive until that cleanup is drained.
            release.store(true, std::memory_order_release);
            service->QuiesceBackend();
        }
    } heldRelease{release, activeClose};
    auto activeTicket = activeClose->TryReserve(); CHECK(activeTicket);
    auto activeSignal = activeTicket->MakeSignal();
    CHECK(activeTicket->Retire(std::make_unique<HeldDestroyPayload>(
        activeCounts, entered, release, heldTimedOut)));
    activeSignal.SignalSuccess();
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    CHECK(entered.load(std::memory_order_acquire));
    std::thread closing([&] { activeClose->QuiesceBackend(); quiesceReturned.store(true, std::memory_order_release); });
    struct CloseGuard {
        std::atomic<bool>& release;
        std::thread& thread;
        ~CloseGuard() {
            release.store(true, std::memory_order_release);
            if (thread.joinable()) thread.join();
        }
    } closeGuard{release, closing};
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!activeClose->BackendQuiesced() &&
           std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    CHECK(activeClose->BackendQuiesced());
    CHECK(!activeClose->TryReserve());
    CHECK(!quiesceReturned.load(std::memory_order_acquire));
    release.store(true, std::memory_order_release); closing.join();
    CHECK(quiesceReturned.load(std::memory_order_acquire));
    CHECK(!heldTimedOut.load(std::memory_order_acquire));
    CHECK(activeCounts.complete == 1 && activeCounts.destroy == 1 &&
          activeCounts.quarantine == 0 && activeCounts.destruct == 1);

    // Quiesce racing Retire's installing gap waits the credit reserved before
    // Installing is published. Once released, Retire observes close and
    // retains the payload without any backend payload virtual call.
    auto installing = Service(719336, 1); Counts installingCounts;
    auto installingTicket = installing->TryReserve(); CHECK(installingTicket);
    auto installGate = std::make_shared<UsdGenExecutionRetirementInstallTestGate>();
    SetUsdGenExecutionRetirementInstallTestGate(installGate);
    std::atomic<bool> installRetired{false}, installClosed{false};
    std::thread installClose;
    std::thread installer([&, ticket = std::move(*installingTicket)] () mutable {
        installRetired.store(ticket.Retire(std::make_unique<Payload>(installingCounts)),
                             std::memory_order_release);
    });
    struct InstallGuard {
        std::shared_ptr<UsdGenExecutionRetirementInstallTestGate> gate;
        std::thread& installer;
        std::thread& closer;
        ~InstallGuard() {
            gate->release.store(true, std::memory_order_release);
            if (installer.joinable()) installer.join();
            if (closer.joinable()) closer.join();
            SetUsdGenExecutionRetirementInstallTestGate({});
        }
    } installGuard{installGate, installer, installClose};
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!installGate->entered.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    CHECK(installGate->entered.load(std::memory_order_acquire));
    installClose = std::thread([&] {
        installing->QuiesceBackend();
        installClosed.store(true, std::memory_order_release);
    });
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!installing->BackendQuiesced() &&
           std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
    CHECK(installing->BackendQuiesced());
    CHECK(!installClosed.load(std::memory_order_acquire));
    installGate->release.store(true, std::memory_order_release);
    installer.join(); installClose.join();
    CHECK(installRetired.load(std::memory_order_acquire) &&
          !installGate->timedOut.load(std::memory_order_acquire) &&
          installClosed.load(std::memory_order_acquire));
    CHECK(installingCounts.complete == 0 && installingCounts.destroy == 0 &&
          installingCounts.quarantine == 0 && installingCounts.destruct == 0);
    SetUsdGenExecutionRetirementInstallTestGate({});

    // The exported process/DSO gate closes both a preexisting service and
    // future registry creation for that backend.
    auto backend = Service(719337, 1); CHECK(backend);
    QuiesceUsdGenExecutionRetirementBackend(usdGenTestUtils::kTestBackendA);
    CHECK(!IsUsdGenExecutionRetirementBackendOpen(usdGenTestUtils::kTestBackendA));
    CHECK(!backend->TryReserve());
    CHECK(!Service(719336, 1));
    return 0;
}
