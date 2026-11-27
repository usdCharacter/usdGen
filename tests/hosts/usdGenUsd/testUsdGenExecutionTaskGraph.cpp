#include "utils/pxrInterop.h"
#include "usdGen/api/ext/device/executionTaskGraph.h"
#include "usdGen/api/ext/device/executionPlan.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <functional>
#include <limits>
#include <string>
#include <type_traits>

#include "pxr/pxr.h"
#include "pxr/base/tf/token.h"

PXR_NAMESPACE_USING_DIRECTIVE

using namespace usdGen;
using Pipeline = UsdGenExecutionPipeline;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)
template<class P> bool Until(P predicate) {
    auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(3);
    while (!predicate()) { if (std::chrono::steady_clock::now()>=deadline) return false; std::this_thread::yield(); }
    return true;
}

static UsdGenExecutionTaskMetadata HazardTask(uint32_t id, uint64_t authoredOrder)
{
    UsdGenExecutionTaskMetadata task;
    task.id = id;
    task.semanticNode = id;
    task.authoredOrderKey = authoredOrder;
    task.kind = UsdGenExecutionTaskKind::Operator;
    task.key = usdGen::UsdGenNodeKey{uint64_t(id) + 1};
    task.type = TfToken("UsdGenTestOperator");
    task.exclusiveWorkspace = false;
    return task;
}

static bool HasDependency(UsdGenExecutionTaskMetadata const& task,
                          uint32_t predecessor, uint8_t provenance)
{
    for (size_t i = 0; i != task.dependencyProvenance.size(); ++i) {
        auto const& edge = task.dependencyProvenance[i];
        if (edge.predecessor == predecessor &&
            (edge.provenance & provenance) == provenance)
            return std::find(task.dependencies.begin(), task.dependencies.end(),
                             predecessor) != task.dependencies.end();
    }
    return false;
}

static UsdGenExecutionTaskMetadata const* FindTask(
    std::vector<UsdGenExecutionTaskMetadata> const& tasks, uint32_t id)
{
    auto found = std::find_if(tasks.begin(), tasks.end(),
                              [id](auto const& task) { return task.id == id; });
    return found == tasks.end() ? nullptr : &*found;
}

static int TestExecutionEstimateContract()
{
    // Unknown timing is orthogonal to known conservative byte components:
    // zero microseconds is never a calibrated zero-cost claim.
    UsdGenExecutionTaskEstimate unknown;
    CHECK(!unknown.memoryAvailable && !unknown.memoryConservativeUpperBound &&
          !unknown.timeAvailable && unknown.estimatedMicroseconds == 0);
    UsdGenExecutionTaskEstimate known;
    known.steadyBytes = 16;
    known.retainedOutputBytes = 32;
    known.producerRetentionBytes = 8;
    known.scratchPeakBytes = 4;
    known.memoryAvailable = true;
    known.memoryConservativeUpperBound = true;
    CHECK(known.memoryAvailable && known.memoryConservativeUpperBound &&
          !known.timeAvailable && known.estimatedMicroseconds == 0 &&
          known.steadyBytes + known.retainedOutputBytes +
              known.producerRetentionBytes + known.scratchPeakBytes == 60);

    UsdGenExecutionGraphMemoryEstimate graph;
    graph.concurrentPeakBytes = 108;
    graph.immutableSharedInputBytes = 92;
    graph.memoryAvailable = true;
    graph.conservativeUpperBound = true;
    std::vector<UsdGenExecutionTaskMetadata> tasks{HazardTask(0, 0)};
    std::vector<UsdGenExecutionValueMetadata> values;
    UsdGenExecutionPlanMetadata metadata(
        "cpu", 1, UsdGenExecutionPlanShape::SourceRootedUnaryDag,
        {}, std::move(tasks), std::move(values), 0, graph);
    CHECK(metadata.MemoryEstimate().memoryAvailable &&
          metadata.MemoryEstimate().conservativeUpperBound &&
          metadata.MemoryEstimate().concurrentPeakBytes == 108 &&
          metadata.MemoryEstimate().immutableSharedInputBytes == 92);

    // Checked arithmetic is required to fail closed at UINT64_MAX. This is
    // deliberately allocation-free so overflow coverage does not depend on
    // constructing an impossibly large geometry descriptor.
    uint64_t total = 0;
    CHECK(!UsdGenExecutionCheckedBytes::Multiply(
              2, std::numeric_limits<uint64_t>::max(), &total) &&
          total == 0);
    total = std::numeric_limits<uint64_t>::max();
    CHECK(!UsdGenExecutionCheckedBytes::Add(&total, 1) &&
          total == std::numeric_limits<uint64_t>::max());
    return 0;
}

static int TestAutomaticHazardLowering()
{
    // Including and exercising this header from the CPU-only task-graph test
    // is intentional: the public plan vocabulary must not require a CUDA,
    // Metal, or Vulkan synchronization type.
    static_assert(std::is_same<decltype(UsdGenExecutionTaskMetadata::dependencies),
                               std::vector<uint32_t>>::value,
                  "public execution metadata uses backend-neutral task ids");
    static_assert(std::is_same<decltype(UsdGenExecutionDependency::predecessor),
                               uint32_t>::value,
                  "public dependency provenance uses backend-neutral ids");

    std::vector<UsdGenExecutionTaskMetadata> tasks;
    for (uint32_t i = 0; i != 9; ++i)
        tasks.push_back(HazardTask(i, i));
    std::vector<UsdGenExecutionValueMetadata> values;
    values.push_back({0, UsdGenExecutionDataKind::ElementGeometry, 0, 1,
                      UsdGenExecutionValueStorage::JobOwnedImmutable});
    values.push_back({1, UsdGenExecutionDataKind::ElementGeometry,
                      UINT32_MAX, 0,
                      UsdGenExecutionValueStorage::ExternalImmutable});
    values.push_back({2, UsdGenExecutionDataKind::ElementGeometry, 5, 1,
                      UsdGenExecutionValueStorage::ExclusiveWorkspaceVersion});
    values.push_back({3, UsdGenExecutionDataKind::ElementGeometry, 7, 2,
                      UsdGenExecutionValueStorage::ExclusiveWorkspaceVersion});
    values.push_back({4, UsdGenExecutionDataKind::TerminalGeneration, 8, 1,
                      UsdGenExecutionValueStorage::PublishedImmutable});

    auto read = [](UsdGenExecutionDataKind resource, uint32_t value,
                   uint32_t producer) {
        UsdGenExecutionResourceUse use;
        use.resource = resource;
        use.access = UsdGenExecutionResourceAccess::Read;
        use.inputValue = value;
        use.producerTask = producer;
        return use;
    };
    auto write = [](UsdGenExecutionDataKind resource, uint32_t value) {
        UsdGenExecutionResourceUse use;
        use.resource = resource;
        use.access = UsdGenExecutionResourceAccess::Write;
        use.outputValue = value;
        return use;
    };
    auto hazard = [](UsdGenExecutionTaskMetadata& task,
                     UsdGenExecutionResourceHazardAccess access) {
        task.resourceHazards.push_back({
            UsdGenExecutionDataKind::ExecutionWorkspace,
            0, access});
        task.exclusiveWorkspace = access !=
            UsdGenExecutionResourceHazardAccess::Read;
    };

    // Two immutable siblings read one producer value. They must both retain
    // the logical producer edge, but must not be serialized with each other.
    tasks[1].resources.push_back(read(UsdGenExecutionDataKind::ElementGeometry,
                                      0, 0));
    tasks[2].resources.push_back(read(UsdGenExecutionDataKind::ElementGeometry,
                                      0, 0));
    // Explicitly mutable workspace users exercise reader/read, reader/write,
    // and write/write ordering. Reads 3 and 4 may overlap; write 5 waits for
    // both readers; read 6 follows that writer; write 7 waits for writer 5
    // and outstanding reader 6.
    tasks[3].resources.push_back(read(
        UsdGenExecutionDataKind::ElementGeometry, 1, UINT32_MAX));
    tasks[4].resources.push_back(read(
        UsdGenExecutionDataKind::ElementGeometry, 1, UINT32_MAX));
    tasks[5].resources.push_back(write(
        UsdGenExecutionDataKind::ElementGeometry, 2));
    tasks[6].resources.push_back(read(
        UsdGenExecutionDataKind::ElementGeometry, 2, 5));
    tasks[7].resources.push_back(write(
        UsdGenExecutionDataKind::ElementGeometry, 3));
    hazard(tasks[3], UsdGenExecutionResourceHazardAccess::Read);
    hazard(tasks[4], UsdGenExecutionResourceHazardAccess::Read);
    hazard(tasks[5], UsdGenExecutionResourceHazardAccess::Write);
    hazard(tasks[6], UsdGenExecutionResourceHazardAccess::Read);
    hazard(tasks[7], UsdGenExecutionResourceHazardAccess::Write);

    // Publication consumes the declared terminal value (task 0's geometry)
    // but joins every launched task for lifetime safety. The duplicate
    // lifetime declaration for task 0 must coalesce with its semantic edge
    // and retain both provenance bits.
    tasks[8].kind = UsdGenExecutionTaskKind::Publication;
    tasks[8].resources.push_back(read(
        UsdGenExecutionDataKind::ElementGeometry, 0, 0));
    tasks[8].resources.push_back(write(
        UsdGenExecutionDataKind::TerminalGeneration, 4));
    for (uint32_t i = 0; i != 8; ++i)
        tasks[8].declaredDependencies.push_back(
            {i, UsdGenExecutionDependencyLifetimePublicationJoin});

    std::string reason;
    CHECK(UsdGenExecutionDependencyCompiler::Lower(&tasks, values, &reason));
    CHECK(reason.empty());
    CHECK(tasks[1].dependencies == std::vector<uint32_t>{0});
    CHECK(tasks[2].dependencies == std::vector<uint32_t>{0});
    CHECK(tasks[1].dependencyProvenance.size() == 1 &&
          HasDependency(tasks[1], 0,
                        UsdGenExecutionDependencySemanticData));
    CHECK(tasks[2].dependencyProvenance.size() == 1 &&
          HasDependency(tasks[2], 0,
                        UsdGenExecutionDependencySemanticData));
    CHECK(!HasDependency(tasks[1], 2,
                         UsdGenExecutionDependencyResourceHazard) &&
          !HasDependency(tasks[2], 1,
                         UsdGenExecutionDependencyResourceHazard));
    CHECK(tasks[3].dependencies.empty() && tasks[4].dependencies.empty());
    CHECK(tasks[5].dependencies == std::vector<uint32_t>({3, 4}));
    CHECK(HasDependency(tasks[5], 3,
                        UsdGenExecutionDependencyResourceHazard) &&
          HasDependency(tasks[5], 4,
                        UsdGenExecutionDependencyResourceHazard));
    CHECK(tasks[6].dependencies == std::vector<uint32_t>{5});
    CHECK(HasDependency(tasks[6], 5,
                        UsdGenExecutionDependencySemanticData) &&
          HasDependency(tasks[6], 5,
                        UsdGenExecutionDependencyResourceHazard));
    CHECK(tasks[6].dependencyProvenance.size() == 1 &&
          tasks[6].dependencyProvenance.front().predecessor == 5 &&
          tasks[6].dependencyProvenance.front().provenance ==
              (UsdGenExecutionDependencySemanticData |
               UsdGenExecutionDependencyResourceHazard));
    CHECK(tasks[7].dependencies == std::vector<uint32_t>({5, 6}));
    CHECK(HasDependency(tasks[7], 5,
                        UsdGenExecutionDependencyResourceHazard) &&
          HasDependency(tasks[7], 6,
                        UsdGenExecutionDependencyResourceHazard));
    CHECK(tasks[8].dependencies ==
          std::vector<uint32_t>({0, 1, 2, 3, 4, 5, 6, 7}));
    CHECK(tasks[8].dependencyProvenance.size() == 8);
    for (uint32_t i = 0; i != 8; ++i)
        CHECK(tasks[8].dependencyProvenance[i].predecessor == i);
    CHECK(HasDependency(tasks[8], 0,
                        UsdGenExecutionDependencySemanticData) &&
          HasDependency(tasks[8], 0,
                        UsdGenExecutionDependencyLifetimePublicationJoin));
    for (uint32_t i = 1; i != 8; ++i)
        CHECK(HasDependency(tasks[8], i,
                            UsdGenExecutionDependencyLifetimePublicationJoin));

    auto lowered = tasks;
    reason.clear();
    CHECK(UsdGenExecutionDependencyCompiler::Validate(lowered, values, &reason));
    CHECK(reason.empty());

    // A malformed logical value must fail closed with a precise reason.
    auto malformed = tasks;
    malformed[1].resources.front().inputValue = 99;
    reason.clear();
    CHECK(!UsdGenExecutionDependencyCompiler::Lower(&malformed, values, &reason));
    CHECK(reason == "resource read names an invalid logical value");

    // Semantic producer edges may form cycles even when task IDs are dense;
    // lowering must reject them rather than handing a deadlocked DAG to an
    // executor.
    std::vector<UsdGenExecutionTaskMetadata> cycleTasks{
        HazardTask(0, 0), HazardTask(1, 1)};
    std::vector<UsdGenExecutionValueMetadata> cycleValues{
        {0, UsdGenExecutionDataKind::ElementGeometry, 0, 1,
         UsdGenExecutionValueStorage::ExclusiveWorkspaceVersion},
        {1, UsdGenExecutionDataKind::ElementGeometry, 1, 1,
         UsdGenExecutionValueStorage::ExclusiveWorkspaceVersion}};
    cycleTasks[0].resources.push_back({
        UsdGenExecutionDataKind::ElementGeometry,
        UsdGenExecutionResourceAccess::ReadWrite, 1, 1, 0});
    cycleTasks[1].resources.push_back({
        UsdGenExecutionDataKind::ElementGeometry,
        UsdGenExecutionResourceAccess::ReadWrite, 0, 0, 1});
    reason.clear();
    CHECK(!UsdGenExecutionDependencyCompiler::Lower(&cycleTasks, cycleValues,
                                                     &reason));
    CHECK(reason == "semantic/data/publication dependencies contain a cycle");

    // A repeated Read + Write declaration for one explicit workspace is a
    // legal ReadWrite declaration, not an order-dependent conflict. The
    // compiler must fold it before applying the reader/writer rules.
    auto mergedAccess = tasks;
    mergedAccess[3].resourceHazards.clear();
    mergedAccess[3].resourceHazards.push_back({
        UsdGenExecutionDataKind::ExecutionWorkspace,
        0, UsdGenExecutionResourceHazardAccess::Read});
    mergedAccess[3].resourceHazards.push_back({
        UsdGenExecutionDataKind::ExecutionWorkspace,
        0, UsdGenExecutionResourceHazardAccess::Write});
    mergedAccess[3].exclusiveWorkspace = true;
    reason.clear();
    CHECK(UsdGenExecutionDependencyCompiler::Lower(&mergedAccess, values,
                                                    &reason));
    CHECK(reason.empty());

    // The task vector is not an ordering authority. Here a reader (semantic
    // task 1) consumes a value produced by writer task 2, while its authored
    // key is earlier. Hazard lowering must first respect the semantic edge,
    // then orient the explicit mutable-workspace edge in that same direction;
    // sorting only by authored key would manufacture a cycle (1 -> 2 -> 1).
    std::vector<UsdGenExecutionTaskMetadata> shuffled{
        HazardTask(1, 10), HazardTask(0, 30), HazardTask(2, 20)};
    shuffled[0].resources.push_back({
        UsdGenExecutionDataKind::ElementGeometry,
        UsdGenExecutionResourceAccess::Read, 2, 1, UINT32_MAX});
    shuffled[2].resources.push_back({
        UsdGenExecutionDataKind::ElementGeometry,
        UsdGenExecutionResourceAccess::Write, UINT32_MAX, UINT32_MAX, 1});
    hazard(shuffled[0], UsdGenExecutionResourceHazardAccess::Read);
    hazard(shuffled[2], UsdGenExecutionResourceHazardAccess::Write);
    std::vector<UsdGenExecutionValueMetadata> shuffledValues{
        {1, UsdGenExecutionDataKind::ElementGeometry, 2, 1,
         UsdGenExecutionValueStorage::ExclusiveWorkspaceVersion}};
    reason.clear();
    CHECK(UsdGenExecutionDependencyCompiler::Lower(&shuffled,
                                                    shuffledValues, &reason));
    auto const* shuffledReader = FindTask(shuffled, 1);
    auto const* shuffledWriter = FindTask(shuffled, 2);
    CHECK(shuffledReader && shuffledWriter);
    CHECK(shuffledReader->dependencies == std::vector<uint32_t>{2});
    CHECK(HasDependency(*shuffledReader, 2,
                        UsdGenExecutionDependencySemanticData) &&
          HasDependency(*shuffledReader, 2,
                        UsdGenExecutionDependencyResourceHazard));
    CHECK(shuffledWriter->dependencies.empty());

    // DataKind is not a physical identity. Two independent workspaces of
    // the same kind must not serialize; each reader only follows the writer
    // for its matching identity.
    std::vector<UsdGenExecutionTaskMetadata> identities;
    for (uint32_t i = 0; i != 4; ++i)
        identities.push_back(HazardTask(i, i));
    identities[0].resourceHazards.push_back({
        UsdGenExecutionDataKind::ExecutionWorkspace, 0,
        UsdGenExecutionResourceHazardAccess::Write});
    identities[0].exclusiveWorkspace = true;
    identities[1].resourceHazards.push_back({
        UsdGenExecutionDataKind::ExecutionWorkspace, 1,
        UsdGenExecutionResourceHazardAccess::Write});
    identities[1].exclusiveWorkspace = true;
    identities[2].resourceHazards.push_back({
        UsdGenExecutionDataKind::ExecutionWorkspace, 0,
        UsdGenExecutionResourceHazardAccess::Read});
    identities[3].resourceHazards.push_back({
        UsdGenExecutionDataKind::ExecutionWorkspace, 1,
        UsdGenExecutionResourceHazardAccess::Read});
    reason.clear();
    CHECK(UsdGenExecutionDependencyCompiler::Lower(
        &identities, std::vector<UsdGenExecutionValueMetadata>{}, &reason));
    CHECK(FindTask(identities, 2)->dependencies == std::vector<uint32_t>{0} &&
          FindTask(identities, 3)->dependencies == std::vector<uint32_t>{1});
    CHECK(!HasDependency(*FindTask(identities, 2), 1,
                         UsdGenExecutionDependencyResourceHazard) &&
          !HasDependency(*FindTask(identities, 3), 0,
                         UsdGenExecutionDependencyResourceHazard));

    auto invalidAccess = identities;
    invalidAccess[0].resourceHazards.front().access =
        static_cast<UsdGenExecutionResourceHazardAccess>(255);
    reason.clear();
    CHECK(!UsdGenExecutionDependencyCompiler::Lower(
        &invalidAccess, std::vector<UsdGenExecutionValueMetadata>{}, &reason));
    CHECK(reason == "resource hazard has an invalid access mode");

    auto invalidWorkspaceValue = values;
    invalidWorkspaceValue.push_back({
        99, UsdGenExecutionDataKind::ExecutionWorkspace, UINT32_MAX, 0,
        UsdGenExecutionValueStorage::ExternalImmutable});
    reason.clear();
    CHECK(!UsdGenExecutionDependencyCompiler::Lower(
        &tasks, invalidWorkspaceValue, &reason));
    CHECK(reason == "workspace identity is only valid in a resource hazard");
    return 0;
}

int main() {
    if (TestExecutionEstimateContract() != 0) return 1;
    if (TestAutomaticHazardLowering() != 0) return 1;
    UsdGenExecutionRuntime runtime(4);
    auto graph = UsdGenExecutionTaskGraph::GetOrCreate(runtime, "test", 0, {8, 32, 32, 2});
    CHECK(graph);
    std::atomic<int> ran{0}, completed{0}; bool returned=false, dependentBeforeReturn=false;
    UsdGenExecutionTaskGraph::Job job;
    job.tasks.resize(3);
    job.tasks[0].run = [&](auto const&, auto done) { ++ran; done({}, {}); returned=true; };
    job.tasks[1].dependencies = {0};
    job.tasks[1].run = [&](auto const&, auto done) { if (!returned) dependentBeforeReturn=true; ++ran; done({}, {}); done({}, {}); };
    job.tasks[2].dependencies = {1};
    job.tasks[2].run = [&](auto const&, auto done) { ++ran; done([] {}, {}); };
    job.completion = [&](auto publish, std::exception_ptr error) {
        if (!error && publish) publish();
        ++completed;
    };
    CHECK(graph->Submit(std::move(job)));
    graph->Drain();
    CHECK(ran == 3 && completed == 1 && !dependentBeforeReturn);

    auto invalid = UsdGenExecutionTaskGraph::GetOrCreate(runtime, "invalid", 0);
    CHECK(!UsdGenExecutionTaskGraph::GetOrCreate(
        runtime, "invalid-zero-burst", 0, {8, 32, 32, 2, 0, 8}));
    UsdGenExecutionTaskGraph::Job cycle; cycle.tasks.resize(2);
    cycle.tasks[0].dependencies={1}; cycle.tasks[1].dependencies={0};
    cycle.tasks[0].run=[](auto const&,auto done){done({},{});}; cycle.tasks[1].run=cycle.tasks[0].run;
    CHECK(!invalid->Submit(std::move(cycle)));
    invalid->Shutdown();

    // Two independent roots become ready concurrently; their common final
    // cannot run until both have settled.
    auto parallel = UsdGenExecutionTaskGraph::GetOrCreate(runtime, "parallel", 0, {4, 16, 16, 2});
    std::atomic<int> entered{0}, parallelDone{0};
    UsdGenExecutionTaskGraph::Job fanin; fanin.tasks.resize(3);
    for (int i=0;i<2;++i) fanin.tasks[i].run=[&](auto const&, auto done) {
        ++entered;
        if (!Until([&] { return entered.load()==2; }))
            done({}, std::make_exception_ptr(std::runtime_error("roots did not overlap")));
        else done({}, {});
    };
    fanin.tasks[2].dependencies={0,1}; fanin.tasks[2].run=[&](auto const&,auto done){++parallelDone; done([]{},{});};
    fanin.completion=[](auto publish,std::exception_ptr error){ if(!error && publish) publish(); };
    CHECK(parallel->Submit(std::move(fanin))); parallel->Drain();
    CHECK(entered==2 && parallelDone==1);

    // First failure skips unscheduled dependents and reports one terminal job.
    auto failed = UsdGenExecutionTaskGraph::GetOrCreate(runtime, "failed", 0);
    std::atomic<int> skipped{0}, failedComplete{0}; UsdGenExecutionTaskGraph::Job bad; bad.tasks.resize(2);
    bad.tasks[0].run=[](auto const&,auto done){done({},std::make_exception_ptr(std::runtime_error("fail")));};
    bad.tasks[1].dependencies={0}; bad.tasks[1].run=[&](auto const&,auto done){++skipped;done([]{},{});};
    bad.completion=[&](auto, std::exception_ptr error){if(error) ++failedComplete;};
    CHECK(failed->Submit(std::move(bad))); failed->Drain();
    CHECK(skipped==0 && failedComplete==1);

    // A single-worker arena autonomously starts independent async roots; no
    // caller may enter/Drain it before the externally-held roots terminate.
    UsdGenExecutionRuntime one(1);
    auto held=UsdGenExecutionTaskGraph::GetOrCreate(one,"held",0,{4,16,16,2});
    std::function<void(Pipeline::Publish,std::exception_ptr)> done[2];
    std::atomic<int> started{0}, faninDone{0}, rootsComplete{0}; UsdGenExecutionTaskGraph::Job roots; roots.tasks.resize(3);
    for(int i=0;i<2;++i) roots.tasks[i].run=[&,i](auto const&,auto finish){done[i]=std::move(finish);started.fetch_add(1,std::memory_order_release);};
    roots.tasks[2].dependencies={0,1}; roots.tasks[2].run=[&](auto const&,auto finish){++faninDone;finish([]{},{});}; roots.completion=[&](auto p,std::exception_ptr e){if(!e&&p)p(); ++rootsComplete;};
    CHECK(held->Submit(std::move(roots)));
    std::thread release([&] {
        if (!Until([&] { return started.load(std::memory_order_acquire) == 2; })) {
            std::fprintf(stderr, "FAIL: one-worker async roots did not start\n");
            std::abort(); // callbacks retain this test's local state
        }
        done[0]({}, {});
        done[1]({}, {});
    });
    release.join();
    if (!Until([&] { return rootsComplete.load(std::memory_order_acquire) == 1; })) {
        std::fprintf(stderr, "FAIL: one-worker async roots did not complete\n");
        std::abort(); // callbacks retain this test's local state
    }
    CHECK(faninDone == 1);
    held->Drain(); // permitted after terminal completion; releases retained callbacks.

    // A terminal callback may enqueue a successor onto the same dispatcher.
    // Completion runs outside the dispatcher's bounded worker node so a
    // maxActiveTasks=1 graph cannot strand the successor behind its callback.
    auto reentry = UsdGenExecutionTaskGraph::GetOrCreate(
        one, "completion-reentry", 0, {4, 8, 8, 1});
    std::atomic<int> reentryComplete{0};
    std::atomic<bool> successorAdmitted{false};
    UsdGenExecutionTaskGraph::Job firstReentry;
    firstReentry.tasks.resize(1);
    firstReentry.tasks[0].run = [](auto const&, auto finish) {
        finish([] {}, {});
    };
    firstReentry.completion = [&](auto publish, std::exception_ptr error) {
        if (!error && publish) publish();
        ++reentryComplete;
        UsdGenExecutionTaskGraph::Job successor;
        successor.tasks.resize(1);
        successor.tasks[0].run = [](auto const&, auto finish) {
            finish([] {}, {});
        };
        successor.completion = [&](auto nextPublish, std::exception_ptr nextError) {
            if (!nextError && nextPublish) nextPublish();
            ++reentryComplete;
        };
        successorAdmitted.store(reentry->Submit(std::move(successor)),
                                std::memory_order_release);
    };
    CHECK(reentry->Submit(std::move(firstReentry)));
    reentry->Drain();
    CHECK(successorAdmitted.load(std::memory_order_acquire) &&
          reentryComplete == 2 &&
          reentry->GetAdmissionUsage().jobs == 0 &&
          reentry->GetAdmissionUsage().tasks == 0 &&
          reentry->GetAdmissionUsage().dependencyEdges == 0);

    // A synchronous Done does not release dependents before the Run call ends.
    auto gate=UsdGenExecutionTaskGraph::GetOrCreate(runtime,"gate",0); bool returnGateReturned=false, early=false; UsdGenExecutionTaskGraph::Job ordering; ordering.tasks.resize(2);
    ordering.tasks[0].run=[&](auto const&,auto finish){finish({},{});returnGateReturned=true;}; ordering.tasks[1].dependencies={0}; ordering.tasks[1].run=[&](auto const&,auto finish){early=!returnGateReturned;finish([]{},{});}; ordering.completion=[](auto p,std::exception_ptr e){if(!e&&p)p();};
    CHECK(gate->Submit(std::move(ordering))); gate->Drain(); CHECK(!early);

    // Both wait families reject a callback that would otherwise wait behind itself.
    Pipeline owner(runtime); auto cross=UsdGenExecutionTaskGraph::GetOrCreate(runtime,"cross",0); std::atomic<int> rejected{0}; UsdGenExecutionTaskGraph::Job waits; waits.tasks.resize(1);
    waits.tasks[0].run=[&](auto const&,auto finish){try{owner.Drain();}catch(std::logic_error const&){++rejected;}try{cross->Drain();}catch(std::logic_error const&){++rejected;}finish([]{},{});}; waits.completion=[](auto p,std::exception_ptr e){if(!e&&p)p();};
    CHECK(cross->Submit(std::move(waits))); cross->Drain(); CHECK(rejected==2);

    // Shutdown waits a held running task, skips future work, and reports once.
    auto stopping=UsdGenExecutionTaskGraph::GetOrCreate(runtime,"stopping",0); std::function<void(Pipeline::Publish,std::exception_ptr)> stopDone; std::atomic<bool> stopReady{false}, stopped{false}; std::atomic<int> stopCompletion{0}; UsdGenExecutionTaskGraph::Job stop; stop.tasks.resize(1);
    stop.tasks[0].run=[&](auto const&,auto finish){stopDone=std::move(finish);stopReady.store(true,std::memory_order_release);}; stop.completion=[&](auto,std::exception_ptr){++stopCompletion;};
    CHECK(stopping->Submit(std::move(stop))); CHECK(Until([&]{return stopReady.load(std::memory_order_acquire);})); std::thread shutdown([&]{stopping->Shutdown();stopped=true;}); CHECK(!stopped.load()); stopDone([]{},{}); shutdown.join(); CHECK(stopped && stopCompletion==1);

    auto sameA=UsdGenExecutionTaskGraph::GetOrCreate(runtime,"singleton",0,{2,8,8,1}); auto sameB=UsdGenExecutionTaskGraph::GetOrCreate(runtime,"singleton",0,{2,8,8,1});
    CHECK(sameA && sameA==sameB); CHECK(!UsdGenExecutionTaskGraph::GetOrCreate(runtime,"singleton",0,{3,8,8,1}));
    return 0;
}
