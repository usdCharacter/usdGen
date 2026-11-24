#ifndef USDGEN_EXECUTION_TASK_GRAPH_H
#define USDGEN_EXECUTION_TASK_GRAPH_H

#include "usdGen/executionPipeline.h"

#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace usdGen {

// Shared, backend-neutral ready-task dispatcher.  It does not reserve memory:
// callers remain responsible for actual DeviceBuffer byte admission.
class UsdGenExecutionTaskGraph {
public:
    using Cancellation = UsdGenExecutionPipeline::Cancellation;
    using Publish = UsdGenExecutionPipeline::Publish;
    using TaskCompletion = std::function<void(Publish, std::exception_ptr)>;
    using TaskRun = std::function<void(Cancellation const&, TaskCompletion)>;
    struct Task { std::vector<uint32_t> dependencies; TaskRun run; };
    using Completion = std::function<void(Publish, std::exception_ptr)>;
    enum class RequestClass { Interactive, Background };
    struct Job {
        Cancellation cancellation;
        std::vector<Task> tasks;
        // Only this task's nonempty publication is forwarded. UINT32_MAX
        // selects the final task in task-vector order.
        uint32_t finalTask = UINT32_MAX;
        RequestClass requestClass = RequestClass::Interactive;
        Completion completion;
    };
    struct Limits {
        uint32_t maxJobs = 256, maxTasks = 4096, maxDependencyEdges = 16384,
                 maxActiveTasks = 8, interactiveBurst = 3,
                 backgroundAgingDispatches = 32;
    };
    // Independently sampled admission counters for test/telemetry only. They
    // include reservations under validation, ingress, queued and running work;
    // this is neither
    // an atomic snapshot nor an acceptance acknowledgement or wait API.
    struct AdmissionUsage { uint32_t jobs = 0, tasks = 0, dependencyEdges = 0; };

    // One process-lifetime dispatcher per backend/physical-device key in the
    // usdGen DSO. Its bounded worker pool is independent of any caller's TBB runtime, so
    // queues from different runtimes can share the device without creating a
    // cross-graph progress dependency. The runtime argument is retained for
    // API compatibility and backend-neutral construction symmetry.
    static std::shared_ptr<UsdGenExecutionTaskGraph> GetOrCreate(
        UsdGenExecutionRuntime&, std::string backend, int device, Limits);
    static std::shared_ptr<UsdGenExecutionTaskGraph> GetOrCreate(
        UsdGenExecutionRuntime& runtime, std::string backend, int device) {
        return GetOrCreate(runtime, std::move(backend), device, Limits{});
    }
    ~UsdGenExecutionTaskGraph();
    UsdGenExecutionTaskGraph(UsdGenExecutionTaskGraph const&) = delete;
    UsdGenExecutionTaskGraph& operator=(UsdGenExecutionTaskGraph const&) = delete;

    // Rejects malformed/cyclic/oversize jobs synchronously, before enqueue.
    bool Submit(Job);
    AdmissionUsage GetAdmissionUsage() const noexcept;
    // External compatibility boundary: waits for independently driven ready
    // tasks without cancelling them. Never call from a task or job completion.
    void Drain();
    // Application/DSO teardown only, never a per-groom queue operation;
    // drains and permanently closes this backend/device singleton.
    void Shutdown();
private:
    struct Impl;
    explicit UsdGenExecutionTaskGraph(std::shared_ptr<Impl>);
    std::shared_ptr<Impl> impl_;
};
}
#endif
