#include "usdGen/exec/executionPipeline.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

using namespace usdGen;
using Pipeline = UsdGenExecutionPipeline;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c); return 1; } } while(false)

// Bounded test rendezvous, not production scheduling or mutual exclusion.
template<class Predicate> bool Until(Predicate predicate) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

int main() {
    UsdGenExecutionRuntime runtime(4);
    Pipeline pipeline(runtime);
    {
        Pipeline probe(runtime);
        Pipeline foreign(runtime);
        CHECK(!probe.IsExecutingOwner() && !foreign.IsExecutingOwner());
        bool ownerIdentity = false, preparationIdentity = false, publicationIdentity = false;
        CHECK(probe.PostCommand([&] {
            ownerIdentity = probe.IsExecutingOwner() && !foreign.IsExecutingOwner();
        }));
        probe.Submit([&](auto const&) -> Pipeline::Publish {
            preparationIdentity = Pipeline::IsExecuting() && !probe.IsExecutingOwner() && !foreign.IsExecutingOwner();
            return [&] { publicationIdentity = probe.IsExecutingOwner() && !foreign.IsExecutingOwner(); };
        });
        probe.Drain();
        CHECK(ownerIdentity && preparationIdentity && publicationIdentity);
        CHECK(!probe.IsExecutingOwner());
    }
    std::atomic<int> completions{0}, publications{0};
    std::atomic<uint64_t> published{0};
    std::atomic<bool> orderError{false};
    auto work = [&](Pipeline::Cancellation const& cancel) -> Pipeline::Publish {
        return [&, epoch=cancel.epoch] {
            auto old = published.exchange(epoch);
            if (old >= epoch) orderError = true;
            ++publications;
        };
    };
    auto complete = [&](uint64_t, Pipeline::Outcome, std::exception_ptr) { ++completions; };
    std::vector<std::thread> submitters;
    for (int t=0;t<4;++t) submitters.emplace_back([&] {
        for (int i=0;i<32;++i) pipeline.Submit(work,complete);
    });
    for (auto& thread : submitters) thread.join();
    pipeline.Drain();
    CHECK(completions == 128 && publications > 0 && !orderError);
    CHECK(published == pipeline.AcceptedEpoch() && published == 128);
    std::vector<int> commands;
    for (int i=0;i<100;++i) CHECK(pipeline.PostCommand([&,i] { commands.push_back(i); }));
    pipeline.Drain();
    CHECK(commands.size() == 100 && pipeline.AcceptedEpoch() == 128);
    for (int i=0;i<100;++i) CHECK(commands[i] == i);

    // A new command is accepted while an older work item is still executing.
    std::atomic<bool> started{false}, observedCancellation{false}, timedOut{false};
    std::atomic<int> stalePublications{0}, superseded{0};
    pipeline.Submit([&](Pipeline::Cancellation const& cancel) -> Pipeline::Publish {
        started = true;
        observedCancellation = Until([&] { return cancel.Superseded(); });
        timedOut = !observedCancellation.load();
        return [&] { ++stalePublications; };
    }, [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr) {
        if (outcome == Pipeline::Outcome::Superseded) ++superseded;
    });
    CHECK(Until([&] { return started.load(); }));
    auto newer = pipeline.Submit(work);
    pipeline.Drain();
    CHECK(!timedOut && observedCancellation && superseded == 1 && stalePublications == 0);
    CHECK(published == newer);

    // Two independently owned descriptions actually execute simultaneously.
    Pipeline other(runtime);
    std::atomic<int> rendezvous{0}, parallelPublished{0};
    auto parallelWork = [&](Pipeline::Cancellation const&) -> Pipeline::Publish {
        ++rendezvous;
        if (!Until([&] { return rendezvous.load() == 2; })) timedOut = true;
        return [&] { ++parallelPublished; };
    };
    pipeline.Submit(parallelWork);
    other.Submit(parallelWork);
    pipeline.Drain(); other.Drain();
    CHECK(!timedOut && parallelPublished == 2);

    // A callback schedules later work without waiting or recursively cooking.
    bool rejectedWait = false;
    int childPublications = 0;
    pipeline.Submit([&](auto const&) -> Pipeline::Publish {
        return [&] {
            try { pipeline.Drain(); }
            catch (std::logic_error const&) { rejectedWait = true; }
            pipeline.Submit([&](auto const&) -> Pipeline::Publish {
                return [&] { ++childPublications; };
            });
        };
    });
    pipeline.Drain();
    CHECK(rejectedWait && childPublications == 1);

    // The terminal publication is the mutable work-to-owner handoff.  A
    // reentrant successor submitted by that publication must remain pending
    // until the closure has completely returned; otherwise SessionCooker's
    // next _Prepare can race the previous cache/snapshot action.  The held
    // closure makes this ordering observable without relying on a final Drain.
    Pipeline publicationOrdered(runtime);
    std::atomic<bool> publicationEntered{false}, releasePublication{false};
    std::atomic<bool> successorEntered{false}, successorEnteredEarly{false};
    CHECK(publicationOrdered.Submit(
        [&](auto const&) -> Pipeline::Publish {
            return [&] {
                publicationEntered.store(true, std::memory_order_release);
                publicationOrdered.Submit([&](auto const&) -> Pipeline::Publish {
                    successorEntered.store(true, std::memory_order_release);
                    if (!releasePublication.load(std::memory_order_acquire))
                        successorEnteredEarly.store(true, std::memory_order_release);
                    return [] {};
                });
                // This rendezvous gives an incorrectly pre-started worker a
                // deterministic opportunity to enter while publication is
                // held.  With the work lane still occupied it cannot enter.
                auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(100);
                while (!successorEntered.load(std::memory_order_acquire) &&
                       std::chrono::steady_clock::now() < deadline)
                    std::this_thread::yield();
                releasePublication.store(true, std::memory_order_release);
            };
        }) != 0);
    publicationOrdered.Drain();
    CHECK(publicationEntered && successorEntered && !successorEnteredEarly);

    int failures = 0;
    pipeline.Submit([](auto const&) -> Pipeline::Publish {
        throw std::runtime_error("intentional work failure");
    }, [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr error) {
        if (outcome == Pipeline::Outcome::Failed && error) ++failures;
        throw std::runtime_error("intentional completion failure");
    });
    pipeline.Drain();
    CHECK(failures == 1 && pipeline.CallbackFailures() == 1);
    CHECK(pipeline.Submit(work) != 0); pipeline.Drain();

    started = false; observedCancellation = false;
    pipeline.Submit([&](Pipeline::Cancellation const& cancel) -> Pipeline::Publish {
        started = true;
        observedCancellation = Until([&] { return cancel.Superseded(); });
        return [&] { ++stalePublications; };
    });
    CHECK(Until([&] { return started.load(); }));
    auto cancellation = pipeline.CancelPending(); pipeline.Drain();
    CHECK(observedCancellation && pipeline.AcceptedEpoch() == cancellation && stalePublications == 0);

    // Shutdown invalidates in-flight work before draining; its captured
    // resources stay alive until the work/completion messages retire.
    auto shutdown = std::make_unique<Pipeline>(runtime);
    started = false; observedCancellation = false;
    std::atomic<int> shutdownCompletions{0};
    shutdown->Submit([&](Pipeline::Cancellation const& cancel) -> Pipeline::Publish {
        started = true;
        observedCancellation = Until([&] { return cancel.Superseded(); });
        return [&] { ++stalePublications; };
    }, [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr) {
        if (outcome == Pipeline::Outcome::Superseded) ++shutdownCompletions;
    });
    CHECK(Until([&] { return started.load(); }));
    shutdown.reset();
    CHECK(observedCancellation && stalePublications == 0 && shutdownCompletions == 1);

    // Shutdown observation sees genuinely closed ingress. A throwing observer
    // cannot bypass draining the asynchronously held work or terminal reply.
    Pipeline observedShutdown(runtime, 2, 2);
    std::promise<Pipeline::AsyncCompletion> readyToClose;
    std::atomic<int> observedTerminals{0};
    CHECK(observedShutdown.SubmitAsync([&](auto const&, auto done) {
        readyToClose.set_value(std::move(done));
    }, [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr) {
        if (outcome == Pipeline::Outcome::Superseded) ++observedTerminals;
    }) != 0);
    auto readyToCloseFuture = readyToClose.get_future();
    CHECK(readyToCloseFuture.wait_for(std::chrono::seconds(5)) == std::future_status::ready);
    auto closeDone = readyToCloseFuture.get();
    auto reservedBeforeClose = observedShutdown.ReserveCommandTicket();
    CHECK(reservedBeforeClose);
    bool closedIngress = false;
    observedShutdown.Shutdown([&] {
        closedIngress = !observedShutdown.ReserveCommandTicket() &&
            !observedShutdown.PostCommand(std::move(reservedBeforeClose), [] {});
        reservedBeforeClose = {};
        closeDone([] {}, {});
        throw std::runtime_error("intentional shutdown observation failure");
    });
    closeDone = {};
    CHECK(closedIngress && observedTerminals == 1);
    CHECK(observedShutdown.OutstandingCommands() == 0);
    CHECK(observedShutdown.CallbackFailures() == 1);

    // Concurrent external compatibility callers each own a reply graph;
    // none concurrently wait on the shared execution graph.
    int ownerValue = 0;
    std::atomic<int> replyFailures{0};
    submitters.clear();
    for (int t=0;t<8;++t) submitters.emplace_back([&] {
        for (int i=0;i<32;++i) {
            int mine = 0;
            pipeline.InvokeOwner([&] { mine = ++ownerValue; });
            if (mine <= 0) ++replyFailures;
        }
    });
    for (auto& thread : submitters) thread.join();
    CHECK(ownerValue == 256 && replyFailures == 0);
    bool commandError = false;
    try { pipeline.InvokeOwner([] { throw std::runtime_error("owner command failure"); }); }
    catch (std::runtime_error const&) { commandError = true; }
    CHECK(commandError);
    pipeline.Await([](auto done) { done(); done(); }); // immediate/idempotent
    bool dispatchError = false;
    try { pipeline.Await([](auto) { throw std::runtime_error("dispatch failed before acceptance"); }); }
    catch (std::runtime_error const&) { dispatchError = true; }
    CHECK(dispatchError);
    int rejectedReplies = 0;
    pipeline.PostCommand([&] {
        try { pipeline.InvokeOwner([] {}); }
        catch (std::logic_error const&) { ++rejectedReplies; }
        try { pipeline.Await([](auto done) { done(); }); }
        catch (std::logic_error const&) { ++rejectedReplies; }
    });
    pipeline.Drain(); CHECK(rejectedReplies == 2);

    // Invalidation from an owner command takes effect during that command,
    // not behind an already queued Finished message.
    uint64_t invalidated = 0;
    pipeline.InvokeOwner([&] {
        invalidated = pipeline.CancelPending();
        if (pipeline.AcceptedEpoch() != invalidated) ++replyFailures;
    });
    CHECK(invalidated && replyFailures == 0);

    // The pipeline retains only running work plus its newest pending request.
    // Eviction completion can itself submit; state is installed before that
    // callback so the reentrant request remains the final pending work.
    Pipeline coalesced(runtime, 3);
    std::atomic<bool> coalescedStarted{false}, coalescedCancelled{false};
    std::atomic<bool> releaseCoalesced{false};
    std::atomic<int> firstPendingRan{0}, firstPendingSuperseded{0};
    std::atomic<int> replacementSuperseded{0}, reentrantPublished{0};
    const auto firstCoalesced = coalesced.Submit([&](Pipeline::Cancellation const& cancel) -> Pipeline::Publish {
        coalescedStarted = true;
        coalescedCancelled = Until([&] { return cancel.Superseded(); });
        Until([&] { return releaseCoalesced.load(); });
        return [] {};
    });
    CHECK(Until([&] { return coalescedStarted.load(); }));
    CHECK(Until([&] { return coalesced.AcceptedEpoch() == firstCoalesced; }));
    CHECK(coalesced.Submit([&](auto const&) -> Pipeline::Publish {
        ++firstPendingRan;
        return [] {};
    }, [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr) {
        if (outcome == Pipeline::Outcome::Superseded) {
            ++firstPendingSuperseded;
            coalesced.Submit([&](auto const&) -> Pipeline::Publish {
                return [&] { ++reentrantPublished; };
            });
        }
    }) != 0);
    CHECK(coalesced.Submit([](auto const&) -> Pipeline::Publish {
        return [] {};
    }, [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr) {
        if (outcome == Pipeline::Outcome::Superseded) ++replacementSuperseded;
    }) != 0);
    CHECK(Until([&] { return firstPendingSuperseded.load() == 1; }));
    releaseCoalesced = true;
    coalesced.Drain();
    CHECK(coalescedCancelled && firstPendingRan == 0 && firstPendingSuperseded == 1);
    CHECK(replacementSuperseded == 1 && reentrantPublished == 1);

    // Credits bound request ingress before messages enter the owner queue.
    Pipeline capped(runtime, 2);
    std::atomic<bool> cappedStarted{false}, cappedCancelled{false};
    std::atomic<bool> releaseCapped{false};
    capped.Submit([&](Pipeline::Cancellation const& cancel) -> Pipeline::Publish {
        cappedStarted = true;
        cappedCancelled = Until([&] { return cancel.Superseded(); });
        Until([&] { return releaseCapped.load(); });
        return [] {};
    });
    CHECK(Until([&] { return cappedStarted.load(); }));
    const auto cappedPending = capped.Submit([](auto const&) -> Pipeline::Publish {
        return [] {};
    });
    CHECK(cappedPending != 0);
    capped.InvokeOwner([] {}); // owner barrier: the second request is pending.
    CHECK(capped.Submit([](auto const&) -> Pipeline::Publish { return [] {}; }) == 0);
    CHECK(capped.AcceptedEpoch() == cappedPending);
    const auto cappedCancel = capped.CancelPending();
    capped.InvokeOwner([] {}); // owner barrier: cancellation reached the running work.
    CHECK(capped.AcceptedEpoch() == cappedCancel);
    releaseCapped = true;
    capped.Drain();
    CHECK(cappedCancelled);

    // An eviction completion may invalidate the replacement immediately. Both
    // pending payloads retire once; the in-flight work observes that cancel.
    Pipeline cancelFromEviction(runtime, 3);
    std::atomic<bool> evictionStarted{false}, evictionCancelled{false};
    std::atomic<bool> releaseEviction{false};
    std::atomic<int> evictedOnce{0}, cancelledReplacementOnce{0}, cancelledRan{0};
    cancelFromEviction.Submit([&](Pipeline::Cancellation const& cancel) -> Pipeline::Publish {
        evictionStarted = true;
        evictionCancelled = Until([&] { return cancel.Superseded(); });
        Until([&] { return releaseEviction.load(); });
        return [] {};
    });
    CHECK(Until([&] { return evictionStarted.load(); }));
    const auto evictedPending = cancelFromEviction.Submit(
        [](auto const&) -> Pipeline::Publish { return [] {}; },
        [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr) {
            if (outcome == Pipeline::Outcome::Superseded) {
                ++evictedOnce;
                cancelFromEviction.CancelPending();
            }
        });
    CHECK(evictedPending != 0);
    CHECK(Until([&] { return cancelFromEviction.AcceptedEpoch() == evictedPending; }));
    CHECK(cancelFromEviction.Submit([&](auto const&) -> Pipeline::Publish {
        ++cancelledRan;
        return [] {};
    }, [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr) {
        if (outcome == Pipeline::Outcome::Superseded) ++cancelledReplacementOnce;
    }) != 0);
    CHECK(Until([&] {
        return evictedOnce.load() == 1 && cancelledReplacementOnce.load() == 1;
    }));
    releaseEviction = true;
    cancelFromEviction.Drain();
    CHECK(evictionCancelled && cancelledRan == 0);

    // Retiring a failed request releases the sole credit before its completion
    // runs, so completion reentry can submit the next request at capacity one.
    Pipeline retryAfterFailure(runtime, 1);
    std::atomic<int> failedOnce{0}, retryPublished{0};
    std::atomic<uint64_t> retryEpoch{0};
    CHECK(retryAfterFailure.Submit([](auto const&) -> Pipeline::Publish {
        throw std::runtime_error("capacity-one failure");
    }, [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr error) {
        if (outcome == Pipeline::Outcome::Failed && error) ++failedOnce;
        retryEpoch = retryAfterFailure.Submit([&](auto const&) -> Pipeline::Publish {
            return [&] { ++retryPublished; };
        });
    }) != 0);
    retryAfterFailure.Drain();
    CHECK(failedOnce == 1 && retryEpoch != 0 && retryPublished == 1);

    // An externally completed stage retains its TBB wait hold without
    // occupying a worker; duplicate and post-terminal callbacks are harmless.
    auto async = std::make_unique<Pipeline>(runtime, 2);
    std::function<void(Pipeline::Publish, std::exception_ptr)> asyncDone;
    std::atomic<bool> asyncReady{false};
    std::atomic<int> asyncPublished{0}, asyncCompleted{0};
    CHECK(async->SubmitAsync([&](Pipeline::Cancellation const&, auto done) {
        asyncDone = std::move(done);
        asyncReady.store(true, std::memory_order_release);
    }, [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr error) {
        if (outcome == Pipeline::Outcome::Published && !error) ++asyncCompleted;
    }) != 0);
    CHECK(Until([&] { return asyncReady.load(std::memory_order_acquire); }));
    asyncDone([&] { ++asyncPublished; }, {});
    asyncDone([&] { ++asyncPublished; }, {});
    async->Drain();
    CHECK(asyncPublished == 1 && asyncCompleted == 1);
    async.reset();
    asyncDone([&] { ++asyncPublished; }, {}); // late after destruction
    CHECK(asyncPublished == 1);

    // An async terminal callback may submit another async request. Drain must
    // include that reentrant successor rather than stranding a graph wait.
    Pipeline asyncReentry(runtime, 2);
    std::atomic<int> asyncReentryCompleted{0};
    std::atomic<bool> asyncReentryError{false};
    std::atomic<uint64_t> asyncReentrySuccessor{0};
    CHECK(asyncReentry.SubmitAsync(
        [](Pipeline::Cancellation const&, auto done) {
            done([] {}, {});
        },
        [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr error) {
            if (outcome != Pipeline::Outcome::Published || error)
                asyncReentryError = true;
            ++asyncReentryCompleted;
            asyncReentrySuccessor = asyncReentry.SubmitAsync(
                [](Pipeline::Cancellation const&, auto done) {
                    done([] {}, {});
                },
                [&](uint64_t, Pipeline::Outcome successorOutcome,
                    std::exception_ptr successorError) {
                    if (successorOutcome != Pipeline::Outcome::Published ||
                        successorError)
                        asyncReentryError = true;
                    ++asyncReentryCompleted;
                });
        }) != 0);
    asyncReentry.Drain();
    CHECK(asyncReentrySuccessor != 0 && asyncReentryCompleted == 2 &&
          !asyncReentryError);

    // A retained late completion token is intentionally small after terminal
    // delivery: it must not keep the original AsyncWork capture alive.
    auto retention=std::make_unique<Pipeline>(runtime);
    auto sentinel=std::make_shared<int>(7); std::weak_ptr<int> weakSentinel=sentinel;
    std::function<void(Pipeline::Publish,std::exception_ptr)> late;
    std::atomic<bool> lateReady{false};
    CHECK(retention->SubmitAsync([keep=sentinel,&late,&lateReady](Pipeline::Cancellation const&,auto done) {
        late=std::move(done); lateReady.store(true,std::memory_order_release);
    }) != 0);
    sentinel.reset(); CHECK(Until([&]{return lateReady.load(std::memory_order_acquire);}));
    late([]{},{}); retention->Drain(); retention.reset();
    CHECK(weakSentinel.expired()); late([]{},{});

    int asyncThrowFailures = 0;
    CHECK(pipeline.SubmitAsync([](Pipeline::Cancellation const&, auto done) {
        done([] {}, {}); throw std::runtime_error("after async completion");
    }, [&](uint64_t, Pipeline::Outcome outcome, std::exception_ptr error) {
        if (outcome == Pipeline::Outcome::Published && !error) ++asyncThrowFailures;
    }) != 0);
    pipeline.Drain(); CHECK(asyncThrowFailures == 1);

    // Destruction is an external boundary: it cooperatively waits for the
    // held backend completion rather than blocking a TBB worker.
    auto held = std::make_unique<Pipeline>(runtime);
    std::function<void(Pipeline::Publish, std::exception_ptr)> heldDone;
    std::atomic<bool> heldReady{false}, destroyStarted{false}, destroyed{false};
    CHECK(held->SubmitAsync([&](Pipeline::Cancellation const&, auto done) {
        heldDone=std::move(done); heldReady.store(true,std::memory_order_release);
    }) != 0);
    CHECK(Until([&] { return heldReady.load(std::memory_order_acquire); }));
    std::thread destroyer([&] { destroyStarted=true; held.reset(); destroyed=true; });
    CHECK(Until([&] { return destroyStarted.load(); }));
    CHECK(!destroyed.load()); heldDone([] {}, {}); destroyer.join(); CHECK(destroyed.load());

    // Command ingress is bounded before callbacks enter TBB's owner queue.
    Pipeline commandCapped(runtime, 2, 1);
    std::atomic<bool> commandStarted{false}, releaseCommand{false};
    std::atomic<int> commandRuns{0};
    CHECK(commandCapped.PostCommand([&] {
        commandStarted = true;
        Until([&] { return releaseCommand.load(); });
        ++commandRuns;
    }));
    CHECK(Until([&] { return commandStarted.load(); }));
    CHECK(commandCapped.PostCommand([&] { ++commandRuns; }));
    auto rejectedCapture = std::make_shared<int>(11);
    std::weak_ptr<int> rejectedWeak = rejectedCapture;
    CHECK(!commandCapped.PostCommand([keep=rejectedCapture] {}));
    rejectedCapture.reset();
    CHECK(rejectedWeak.expired());
    CHECK(commandCapped.OutstandingCommands() == 1 && commandCapped.CommandCapacity() == 1);
    releaseCommand = true;
    commandCapped.Drain();
    CHECK(commandRuns == 2 && commandCapped.OutstandingCommands() == 0);

    // Reservation before producer handoff lets an admitted lifecycle reply
    // complete when ordinary public command ingress is full.
    Pipeline ticketed(runtime, 2, 1);
    auto lifecycleTicket = ticketed.ReserveCommandTicket();
    CHECK(lifecycleTicket && ticketed.OutstandingCommands() == 1);
    CHECK(!ticketed.PostCommand([] {}));
    std::atomic<int> ticketRuns{0};
    CHECK(ticketed.PostCommand(std::move(lifecycleTicket), [&] { ++ticketRuns; }));
    ticketed.Drain();
    CHECK(ticketRuns == 1 && ticketed.OutstandingCommands() == 0);

    // A durable mailbox holds one bounded credit and coalesces producer
    // updates while its owner callback is executing.
    Pipeline mailboxPipeline(runtime, 2, 1);
    auto mailbox = mailboxPipeline.ReserveCommandMailbox();
    CHECK(mailbox && !mailboxPipeline.PostCommand([] {}));
    std::atomic<bool> mailboxStarted{false}, releaseMailbox{false};
    std::atomic<int> mailboxValue{0};
    CHECK(mailboxPipeline.PostLatestCommand(mailbox, [&] {
        mailboxStarted = true;
        Until([&] { return releaseMailbox.load(); });
        mailboxValue = 1;
    }));
    CHECK(Until([&] { return mailboxStarted.load(); }));
    CHECK(mailboxPipeline.PostLatestCommand(mailbox, [&] { mailboxValue = 2; }));
    releaseMailbox = true;
    mailboxPipeline.Drain();
    CHECK(mailboxValue == 2 && mailboxPipeline.OutstandingCommands() == 1);
    mailbox = {};
    CHECK(mailboxPipeline.OutstandingCommands() == 0);

    // Concurrent producers coalesce behind a running owner callback without
    // consuming another public command credit. Replaced immutable captures do
    // not remain retained after the final handoff drains.
    Pipeline concurrentMailbox(runtime, 2, 1);
    auto concurrentBox = concurrentMailbox.ReserveCommandMailbox();
    CHECK(concurrentBox && !concurrentMailbox.PostCommand([] {}));
    std::atomic<bool> concurrentStarted{false}, releaseConcurrent{false};
    std::atomic<int> concurrentValue{0}, concurrentPosts{0};
    CHECK(concurrentMailbox.PostLatestCommand(concurrentBox, [&] {
        concurrentStarted = true;
        Until([&] { return releaseConcurrent.load(); });
    }));
    CHECK(Until([&] { return concurrentStarted.load(); }));
    std::vector<std::shared_ptr<int>> replacements;
    std::vector<std::weak_ptr<int>> replacementWeak;
    for (int i=0; i<64; ++i) {
        replacements.push_back(std::make_shared<int>(i));
        replacementWeak.push_back(replacements.back());
    }
    submitters.clear();
    for (int i=0; i<64; ++i) submitters.emplace_back([&, i] {
        auto keep = replacements[i];
        if (concurrentMailbox.PostLatestCommand(concurrentBox, [&, keep, i] {
                concurrentValue = i;
            })) ++concurrentPosts;
    });
    for (auto& thread : submitters) thread.join();
    CHECK(concurrentPosts == 64 && concurrentMailbox.OutstandingCommands() == 1);
    CHECK(concurrentMailbox.PostLatestCommand(concurrentBox, [&] { concurrentValue = 999; }));
    replacements.clear();
    releaseConcurrent = true;
    concurrentMailbox.Drain();
    CHECK(concurrentValue == 999 && concurrentMailbox.OutstandingCommands() == 1);
    for (auto const& weak : replacementWeak) CHECK(weak.expired());
    concurrentBox = {};
    CHECK(concurrentMailbox.OutstandingCommands() == 0);

    // A command releases its credit before invoking user code, so a cap-one
    // callback can post its successor without recursive waiting.
    Pipeline commandReentry(runtime, 2, 1);
    std::atomic<int> reentrantCommands{0}; std::atomic<bool> reentrantPost{false};
    CHECK(commandReentry.PostCommand([&] {
        ++reentrantCommands;
        reentrantPost = commandReentry.PostCommand([&] { ++reentrantCommands; });
    }));
    commandReentry.Drain();
    CHECK(reentrantPost && reentrantCommands == 2 && commandReentry.OutstandingCommands() == 0);

    // A ticket rejected by another pipeline remains owned by its origin and
    // can still deliver exactly once there.
    Pipeline ticketOrigin(runtime, 2, 1), ticketWrong(runtime, 2, 1);
    auto originTicket = ticketOrigin.ReserveCommandTicket();
    std::atomic<int> originRuns{0};
    CHECK(originTicket && !ticketWrong.PostCommand(std::move(originTicket), [] {}));
    CHECK(ticketOrigin.OutstandingCommands() == 1 && ticketWrong.OutstandingCommands() == 0);
    CHECK(ticketOrigin.PostCommand(std::move(originTicket), [&] { ++originRuns; }));
    ticketOrigin.Drain();
    CHECK(originRuns == 1 && ticketOrigin.OutstandingCommands() == 0);

    // Shutdown discards a queued command, runs its recovery exactly once, and
    // releases its admission before that recovery observes the pipeline.
    auto cancelRecovery = std::make_unique<Pipeline>(runtime, 2, 2);
    std::atomic<bool> recoveryStarted{false}, releaseRecovery{false};
    std::atomic<int> recovered{0}; std::atomic<uint64_t> recoveryCredits{UINT64_MAX};
    CHECK(cancelRecovery->PostCommand([&] {
        recoveryStarted = true;
        Until([&] { return releaseRecovery.load(); });
    }));
    CHECK(Until([&] { return recoveryStarted.load(); }));
    CHECK(cancelRecovery->PostCommand([] {}, [&] {
        recoveryCredits = cancelRecovery->OutstandingCommands(); ++recovered;
    }));
    std::thread recoveryShutdown([&] { cancelRecovery->Shutdown(); });
    CHECK(Until([&] {
        auto probe = cancelRecovery->ReserveCommandTicket();
        return !probe;
    }));
    releaseRecovery = true;
    recoveryShutdown.join();
    CHECK(recovered == 1 && recoveryCredits == 0 && cancelRecovery->OutstandingCommands() == 0);

    // Work terminals retain their existing request credit, so a Finished
    // message cannot be rejected just because public command ingress is full.
    Pipeline terminalLane(runtime, 2, 1);
    std::function<void(Pipeline::Publish, std::exception_ptr)> terminalDone;
    std::atomic<bool> terminalReady{false}, ownerHeld{false}, releaseOwner{false};
    std::atomic<int> terminalPublished{0};
    CHECK(terminalLane.SubmitAsync([&](Pipeline::Cancellation const&, auto done) {
        terminalDone = std::move(done); terminalReady = true;
    }) != 0);
    CHECK(Until([&] { return terminalReady.load(); }));
    CHECK(terminalLane.PostCommand([&] {
        ownerHeld = true;
        Until([&] { return releaseOwner.load(); });
    }));
    CHECK(Until([&] { return ownerHeld.load(); }));
    CHECK(terminalLane.PostCommand([] {}));
    CHECK(!terminalLane.PostCommand([] {}));
    terminalDone([&] { ++terminalPublished; }, {});
    releaseOwner = true;
    terminalLane.Drain();
    CHECK(terminalPublished == 1);

    // Public cancellations are bounded; owner reentry invalidates inline even
    // while a queued command owns the sole admission credit.
    Pipeline cancelIngress(runtime, 2, 1);
    std::atomic<bool> cancelOwnerStarted{false}, allowInlineCancel{false};
    std::atomic<uint64_t> inlineCancel{0};
    CHECK(cancelIngress.PostCommand([&] {
        cancelOwnerStarted = true;
        Until([&] { return allowInlineCancel.load(); });
        inlineCancel = cancelIngress.CancelPending();
    }));
    CHECK(Until([&] { return cancelOwnerStarted.load(); }));
    CHECK(cancelIngress.PostCommand([] {}));
    CHECK(cancelIngress.CancelPending() == 0);
    allowInlineCancel = true;
    cancelIngress.Drain();
    CHECK(inlineCancel != 0 && cancelIngress.AcceptedEpoch() == inlineCancel);

    int completedCommands = 0, cancelledCommands = 0;
    {
        Pipeline closing(runtime);
        for (int i=0;i<128;++i)
            CHECK(closing.PostCommand([&] { ++completedCommands; }, [&] { ++cancelledCommands; }));
    }
    CHECK(completedCommands + cancelledCommands == 128);

    // A native completion producer can resume an idle owner while an external
    // lifecycle boundary drains it. Keep one waiter, but race admission with
    // repeated idle-to-active transitions so the graph root must stay alive.
    {
        constexpr unsigned commandCount = 10000;
        Pipeline concurrentDrain(runtime, 2, 2);
        std::vector<unsigned> commandHits(commandCount, 0);
        std::atomic<unsigned> completed{0}, drainCycles{0};
        std::atomic<bool> producerDone{false}, drainerDone{false}, rejected{false};
        std::thread drainer([&] {
            do {
                concurrentDrain.Drain();
                drainCycles.fetch_add(1, std::memory_order_release);
                std::this_thread::yield();
            } while (!producerDone.load(std::memory_order_acquire));
            concurrentDrain.Drain();
            drainerDone.store(true, std::memory_order_release);
        });
        std::thread producer([&] {
            for (unsigned index = 0; index != commandCount; ++index) {
                // Each command follows a drain and the preceding callback.
                // The waiter continues independently during admission.
                while (drainCycles.load(std::memory_order_acquire) <= index)
                    std::this_thread::yield();
                if (!concurrentDrain.PostCommand([&, index] {
                    ++commandHits[index];
                    completed.fetch_add(1, std::memory_order_release);
                })) {
                    rejected.store(true, std::memory_order_release);
                    break;
                }
                while (completed.load(std::memory_order_acquire) != index + 1)
                    std::this_thread::yield();
            }
            producerDone.store(true, std::memory_order_release);
        });
        if (!Until([&] { return drainerDone.load(std::memory_order_acquire); })) {
            // A damaged root can leave Drain spinning forever. Fail this
            // bounded regression without joining that blocked waiter.
            std::fprintf(stderr, "concurrent admission/Drain exceeded its deadline\n");
            std::abort();
        }
        producer.join();
        drainer.join();
        CHECK(!rejected && completed == commandCount && drainCycles >= commandCount);
        for (auto hits : commandHits) CHECK(hits == 1);
        CHECK(concurrentDrain.OutstandingCommands() == 0);
    }
    return 0;
}
