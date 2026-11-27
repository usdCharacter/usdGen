#include "usdGen/api/ext/device/executionTaskGraph.h"

#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>

using namespace usdGen;
using Graph = UsdGenExecutionTaskGraph;

#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

template <class Predicate>
bool Until(Predicate predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    while (!predicate()) {
        if (std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}

int main() {
    UsdGenExecutionRuntime runtime(4);

    // maxJobs is released before a completion callback. That callback can
    // therefore admit replacement work while the first async task's holder is
    // retired, without a second accepted job ever occupying the graph.
    auto oneJob = Graph::GetOrCreate(runtime, "stress-reentry", 0,
        {1, 8, 8, 1, 2, 8});
    CHECK(oneJob);
    std::shared_ptr<Graph::TaskCompletion> heldCompletion;
    std::atomic<int> firstComplete{0}, replacementComplete{0}, publications{0};
    std::atomic<bool> reentryError{false};
    Graph::Job held;
    held.tasks.resize(1);
    held.tasks[0].run = [&](auto const&, auto done) {
        auto holder = std::make_shared<Graph::TaskCompletion>(std::move(done));
        std::atomic_store_explicit(&heldCompletion, std::move(holder),
                                   std::memory_order_release);
    };
    held.completion = [&](auto publish, std::exception_ptr error) {
        if (error || !publish) reentryError = true;
        else publish();
        ++firstComplete;
        Graph::Job replacement;
        replacement.tasks.resize(1);
        replacement.tasks[0].run = [](auto const&, auto done) {
            done([] {}, {});
        };
        replacement.completion = [&](auto next, std::exception_ptr nextError) {
            if (nextError || !next) reentryError = true;
            else next();
            ++replacementComplete;
        };
        if (!oneJob->Submit(std::move(replacement))) reentryError = true;
    };
    CHECK(oneJob->Submit(std::move(held)));
    CHECK(Until([&] { return std::atomic_load_explicit(&heldCompletion,
        std::memory_order_acquire) != nullptr; }));
    Graph::Job rejected;
    rejected.tasks.resize(1);
    rejected.tasks[0].run = [](auto const&, auto done) { done([] {}, {}); };
    CHECK(!oneJob->Submit(std::move(rejected)));
    auto firstDone = std::atomic_load_explicit(&heldCompletion, std::memory_order_acquire);
    CHECK(firstDone);
    (*firstDone)([&] { ++publications; }, {});
    oneJob->Drain();
    CHECK(!reentryError && firstComplete == 1 && replacementComplete == 1 && publications == 1);

    // Rejected oversize/edge-heavy DAGs must not consume counters needed by a
    // following full-capacity valid submission. Cycle and final-reachability
    // validation take a different rejection route and must leave the same
    // counters available.
    auto bounded = Graph::GetOrCreate(runtime, "stress-limits", 0,
        {8, 3, 2, 1, 2, 8});
    CHECK(bounded);
    Graph::Job tooManyTasks;
    tooManyTasks.tasks.resize(4);
    for (auto& task : tooManyTasks.tasks)
        task.run = [](auto const&, auto done) { done({}, {}); };
    CHECK(!bounded->Submit(std::move(tooManyTasks)));
    Graph::Job tooManyEdges;
    tooManyEdges.tasks.resize(3);
    for (auto& task : tooManyEdges.tasks)
        task.run = [](auto const&, auto done) { done({}, {}); };
    tooManyEdges.tasks[2].dependencies = {0, 1, 0};
    CHECK(!bounded->Submit(std::move(tooManyEdges)));
    Graph::Job cycle;
    cycle.tasks.resize(3);
    for (auto& task : cycle.tasks)
        task.run = [](auto const&, auto done) { done({}, {}); };
    cycle.tasks[0].dependencies = {2};
    cycle.tasks[1].dependencies = {0};
    cycle.tasks[2].dependencies = {1};
    CHECK(!bounded->Submit(std::move(cycle)));
    Graph::Job unreachableFinal;
    unreachableFinal.tasks.resize(3);
    for (auto& task : unreachableFinal.tasks)
        task.run = [](auto const&, auto done) { done({}, {}); };
    unreachableFinal.tasks[2].dependencies = {0}; // task 1 is not an ancestor
    CHECK(!bounded->Submit(std::move(unreachableFinal)));
    std::atomic<int> recovered{0};
    std::atomic<bool> recoveryError{false};
    Graph::Job valid;
    valid.tasks.resize(3);
    valid.tasks[0].run = [](auto const&, auto done) { done({}, {}); };
    valid.tasks[1].dependencies = {0};
    valid.tasks[1].run = [](auto const&, auto done) { done({}, {}); };
    valid.tasks[2].dependencies = {1};
    valid.tasks[2].run = [](auto const&, auto done) { done([] {}, {}); };
    valid.completion = [&](auto publish, std::exception_ptr error) {
        if (error || !publish) recoveryError = true;
        else publish();
        ++recovered;
    };
    CHECK(bounded->Submit(std::move(valid)));
    bounded->Drain();
    CHECK(!recoveryError && recovered == 1);

    // Hold the only active worker, then make both classes ready before
    // releasing it. Every work item is a two-stage chain, so the trace checks
    // scheduling rather than merely admission order.
    auto fair = Graph::GetOrCreate(runtime, "stress-fair", 0,
        {16, 64, 64, 1, 2, 32});
    CHECK(fair);
    std::shared_ptr<Graph::TaskCompletion> sentinelCompletion;
    std::atomic<int> sentinelComplete{0};
    Graph::Job sentinel;
    sentinel.tasks.resize(1);
    sentinel.tasks[0].run = [&](auto const&, auto done) {
        auto holder = std::make_shared<Graph::TaskCompletion>(std::move(done));
        std::atomic_store_explicit(&sentinelCompletion, std::move(holder),
                                   std::memory_order_release);
    };
    sentinel.completion = [&](auto publish, std::exception_ptr error) {
        if (!error && publish) publish();
        ++sentinelComplete;
    };
    CHECK(fair->Submit(std::move(sentinel)));
    CHECK(Until([&] { return std::atomic_load_explicit(&sentinelCompletion,
        std::memory_order_acquire) != nullptr; }));

    constexpr int interactiveJobs = 4;
    constexpr int backgroundJobs = 3;
    std::array<std::atomic<int>, 2 * (interactiveJobs + backgroundJobs)> trace;
    for (auto& item : trace) item.store(-1, std::memory_order_relaxed);
    std::atomic<int> traceSize{0}, interactiveComplete{0}, backgroundComplete{0};
    std::atomic<bool> fairError{false};
    auto submitChain = [&](Graph::RequestClass requestClass) {
        Graph::Job job;
        job.requestClass = requestClass;
        job.tasks.resize(2);
        for (int stage = 0; stage != 2; ++stage) {
            if (stage) job.tasks[stage].dependencies = {0};
            job.tasks[stage].run = [&, requestClass, stage](auto const&, auto done) {
                const int slot = traceSize.fetch_add(1, std::memory_order_relaxed);
                trace[slot].store(requestClass == Graph::RequestClass::Interactive ? 1 : 0,
                                  std::memory_order_release);
                done(stage ? Graph::Publish([] {}) : Graph::Publish{}, {});
            };
        }
        job.completion = [&, requestClass](auto publish, std::exception_ptr error) {
            if (error || !publish) fairError = true;
            else publish();
            if (requestClass == Graph::RequestClass::Interactive) ++interactiveComplete;
            else ++backgroundComplete;
        };
        if (!fair->Submit(std::move(job))) fairError = true;
    };
    for (int i = 0; i != interactiveJobs; ++i) submitChain(Graph::RequestClass::Interactive);
    for (int i = 0; i != backgroundJobs; ++i) submitChain(Graph::RequestClass::Background);
    auto sentinelDone = std::atomic_load_explicit(&sentinelCompletion, std::memory_order_acquire);
    CHECK(sentinelDone);
    (*sentinelDone)([] {}, {});
    fair->Drain();
    CHECK(!fairError && sentinelComplete == 1 &&
          interactiveComplete == interactiveJobs && backgroundComplete == backgroundJobs);
    CHECK(traceSize == static_cast<int>(trace.size()));
    int consecutiveInteractive = 0;
    int observedInteractive = 0, observedBackground = 0;
    for (auto const& item : trace) {
        if (item.load(std::memory_order_acquire)) {
            ++observedInteractive;
            ++consecutiveInteractive;
            CHECK(consecutiveInteractive <= 2);
        } else {
            ++observedBackground;
            consecutiveInteractive = 0;
        }
    }
    CHECK(observedInteractive == 2 * interactiveJobs &&
          observedBackground == 2 * backgroundJobs);

    // Aging must override an otherwise generous interactive burst while
    // interactive work remains ready. The single background task is expected
    // within two service turns after the sentinel releases the worker.
    auto aging = Graph::GetOrCreate(runtime, "stress-aging", 0,
        {16, 32, 32, 1, 8, 1});
    CHECK(aging);
    std::shared_ptr<Graph::TaskCompletion> agingSentinel;
    Graph::Job agingHold;
    agingHold.tasks.resize(1);
    agingHold.tasks[0].run = [&](auto const&, auto done) {
        auto holder = std::make_shared<Graph::TaskCompletion>(std::move(done));
        std::atomic_store_explicit(&agingSentinel, std::move(holder),
                                   std::memory_order_release);
    };
    agingHold.completion = [](auto publish, std::exception_ptr error) {
        if (!error && publish) publish();
    };
    CHECK(aging->Submit(std::move(agingHold)));
    CHECK(Until([&] { return std::atomic_load_explicit(&agingSentinel,
        std::memory_order_acquire) != nullptr; }));
    std::array<std::atomic<int>, 4> agingTrace;
    for (auto& item : agingTrace) item.store(-1, std::memory_order_relaxed);
    std::atomic<int> agingSize{0}, agingInteractive{0}, agingBackground{0};
    std::atomic<bool> agingError{false};
    auto submitAging = [&](Graph::RequestClass requestClass) {
        Graph::Job job;
        job.requestClass = requestClass;
        job.tasks.resize(1);
        job.tasks[0].run = [&, requestClass](auto const&, auto done) {
            const int slot = agingSize.fetch_add(1, std::memory_order_relaxed);
            agingTrace[slot].store(requestClass == Graph::RequestClass::Interactive ? 1 : 0,
                                   std::memory_order_release);
            done([] {}, {});
        };
        job.completion = [&, requestClass](auto publish, std::exception_ptr error) {
            if (error || !publish) agingError = true;
            else publish();
            if (requestClass == Graph::RequestClass::Interactive) ++agingInteractive;
            else ++agingBackground;
        };
        if (!aging->Submit(std::move(job))) agingError = true;
    };
    for (int i = 0; i != 3; ++i) submitAging(Graph::RequestClass::Interactive);
    submitAging(Graph::RequestClass::Background);
    auto agingDone = std::atomic_load_explicit(&agingSentinel, std::memory_order_acquire);
    CHECK(agingDone);
    (*agingDone)([] {}, {});
    aging->Drain();
    CHECK(!agingError && agingSize == 4 && agingInteractive == 3 && agingBackground == 1);
    int backgroundTurn = -1;
    for (int i = 0; i != 4; ++i)
        if (agingTrace[i].load(std::memory_order_acquire) == 0) backgroundTurn = i;
    CHECK(backgroundTurn >= 0 && backgroundTurn < 2);
    bool interactiveAfterBackground = false;
    for (int i = backgroundTurn + 1; i != 4; ++i)
        interactiveAfterBackground |= agingTrace[i].load(std::memory_order_acquire) == 1;
    CHECK(interactiveAfterBackground);
    return 0;
}
