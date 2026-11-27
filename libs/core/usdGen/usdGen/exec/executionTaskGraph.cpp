#include "usdGen/api/ext/device/executionTaskGraph.h"
#include "usdGen/exec/executionRuntimeInternal.h"
#include "usdGen/api/epoch.h"
#include "usdGen/api/taskScheduler.h"

#include <algorithm>
#include <atomic>
#include <optional>
#include <stdexcept>
#include <utility>

namespace usdGen {
namespace {
thread_local bool executingTaskGraph = false;
struct TaskGraphScope {
    bool previous = executingTaskGraph;
    bool previousAny = usdGenExecutionGraphActive;
    TaskGraphScope() {
        executingTaskGraph = true;
        usdGenExecutionGraphActive = true;
    }
    ~TaskGraphScope() {
        executingTaskGraph = previous;
        usdGenExecutionGraphActive = previousAny;
    }
};
struct RegistryKey {
    std::string backend;
    int device = -1;
    bool operator==(RegistryKey const& other) const {
        return device == other.device && backend == other.backend;
    }
};
}

// Dispatcher state has one writer at a time: a lane. Producers push messages
// on a Treiber stack; whoever wins `laneBusy` handles every queued message,
// releases the lane and rechecks, so no message is left behind and nobody waits.
// Ready tasks and completion callbacks run on this graph's own scheduler
// (maxActiveTasks workers). Drain parks on the pending job count.
struct UsdGenExecutionTaskGraph::Impl : std::enable_shared_from_this<Impl> {
    struct State {
        Job job;
        std::vector<uint32_t> remaining;
        std::vector<std::vector<uint32_t>> children;
        std::vector<bool> started;
        uint32_t running = 0;
        uint32_t settled = 0;
        uint32_t edgeCount = 0;
        bool cancelled = false;
        bool completed = false;
        uint64_t lastServed = 0;
        std::exception_ptr error;
        Publish publish;
    };
    struct Message {
        enum class Kind { Add, Finish, Shutdown } kind = Kind::Add;
        Message* next = nullptr;
        std::shared_ptr<State> state;
        uint32_t task = 0;
        Publish publish;
        std::exception_ptr error;
    };
    struct RunTask : UsdGenTask {
        std::shared_ptr<Impl> self;
        std::shared_ptr<State> state;
        uint32_t task = 0;
    };
    struct CallbackTask : UsdGenTask {
        std::shared_ptr<Impl> self;
        Completion callback;
        Publish publish;
        std::exception_ptr error;
        CallbackTask* nextCallback = nullptr;
    };

    explicit Impl(Limits value)
        : limits(value),
          scheduler(std::make_unique<UsdGenTaskScheduler>(int(value.maxActiveTasks))) {
        jobs.reserve(limits.maxJobs);
        shutdownMessage.kind = Message::Kind::Shutdown;
    }
    ~Impl() = default;

    Limits limits;
    std::unique_ptr<UsdGenTaskScheduler> scheduler;
    // Lane-owned.
    std::vector<std::shared_ptr<State>> jobs;
    uint32_t cursor = 0;
    uint32_t active = 0;
    uint64_t dispatchTick = 0;
    uint32_t interactiveStreak = 0;
    bool shutdownSeen = false;
    CallbackTask* deferred = nullptr;
    Message shutdownMessage;  // preallocated, never deleted
    // Shared.
    UsdGenTreiberStack<Message, &Message::next> inbox;
    std::atomic<bool> laneBusy{false};
    UsdGenPendingCount pending;  // admitted jobs until their callback returned
    std::atomic<bool> closing{false};
    std::atomic<bool> shutdownStarted{false};
    std::atomic<uint32_t> admittedJobs{0};
    std::atomic<uint32_t> admittedTasks{0};
    std::atomic<uint32_t> admittedEdges{0};

    void Dispatch() {
        while (active < limits.maxActiveTasks && !jobs.empty()) {
            bool launched = false;
            auto const count = static_cast<uint32_t>(jobs.size());
            bool hasInteractive = false;
            bool hasBackground = false;
            bool agedBackground = false;
            for (auto const& state : jobs) if (!state->cancelled) {
                bool ready = false;
                for (uint32_t i = 0; i != state->remaining.size(); ++i)
                    if (!state->started[i] && state->remaining[i] == 0) {
                        ready = true;
                        break;
                    }
                if (!ready) continue;
                if (state->job.requestClass == RequestClass::Interactive)
                    hasInteractive = true;
                else {
                    hasBackground = true;
                    agedBackground |= dispatchTick - state->lastServed >=
                        limits.backgroundAgingDispatches;
                }
            }
            bool const chooseBackground = hasBackground && (!hasInteractive ||
                (limits.interactiveBurst &&
                 interactiveStreak >= limits.interactiveBurst) ||
                (agedBackground && interactiveStreak > 0));
            for (uint32_t probe = 0; probe != count; ++probe) {
                cursor %= count;
                auto state = jobs[cursor];
                cursor = (cursor + 1) % count;
                if (state->cancelled) continue;
                bool const background =
                    state->job.requestClass == RequestClass::Background;
                if (background != chooseBackground &&
                    (chooseBackground ? hasBackground : hasInteractive))
                    continue;
                for (uint32_t i = 0; i != state->remaining.size(); ++i) {
                    if (state->started[i] || state->remaining[i] != 0) continue;
                    state->started[i] = true;
                    ++state->running;
                    ++active;
                    auto* run = new RunTask;
                    run->taskRun = &RunReady;
                    run->self = shared_from_this();
                    run->state = state;
                    run->task = i;
                    ++dispatchTick;
                    state->lastServed = dispatchTick;
                    if (state->job.requestClass == RequestClass::Interactive) {
                        if (interactiveStreak < limits.interactiveBurst)
                            ++interactiveStreak;
                    } else interactiveStreak = 0;
                    launched = true;
                    scheduler->Post(run);
                    break;
                }
                if (launched) break;
            }
            if (!launched) break;
        }
    }

    void Complete(std::shared_ptr<State> const& state) {
        if (state->completed || state->running != 0) return;
        state->completed = true;
        if (!state->error && !state->publish)
            state->error = std::make_exception_ptr(
                std::runtime_error("final task returned no publication"));
        Completion callback = std::move(state->job.completion);
        Publish publish = std::move(state->publish);
        std::exception_ptr error = state->error;
        uint32_t const taskCount =
            static_cast<uint32_t>(state->job.tasks.size());
        jobs.erase(std::remove(jobs.begin(), jobs.end(), state), jobs.end());
        std::vector<Task>{}.swap(state->job.tasks);
        state->job.cancellation = {};
        std::vector<uint32_t>{}.swap(state->remaining);
        std::vector<std::vector<uint32_t>>{}.swap(state->children);
        std::vector<bool>{}.swap(state->started);
        state->error = {};
        admittedEdges.fetch_sub(state->edgeCount, std::memory_order_acq_rel);
        admittedTasks.fetch_sub(taskCount, std::memory_order_acq_rel);
        admittedJobs.fetch_sub(1, std::memory_order_acq_rel);
        if (!callback) {
            pending.Release();
            return;
        }
        auto* task = new CallbackTask;
        task->taskRun = &RunCallback;
        task->self = shared_from_this();
        task->callback = std::move(callback);
        task->publish = std::move(publish);
        task->error = std::move(error);
        task->nextCallback = deferred;
        deferred = task;
    }

    // Cancels every admitted job: queued tasks settle now, running ones
    // report through their own Finish.
    void CancelAll() {
        for (auto const& state : jobs) {
            state->cancelled = true;
            if (!state->error)
                state->error = std::make_exception_ptr(
                    std::runtime_error("task graph shutdown"));
            for (uint32_t i = 0; i != state->started.size(); ++i)
                if (!state->started[i]) {
                    state->started[i] = true;
                    ++state->settled;
                }
        }
        for (;;) {
            auto found = std::find_if(jobs.begin(), jobs.end(),
                [](auto const& state) {
                    return state->cancelled && !state->completed &&
                        state->running == 0 &&
                        state->settled == state->started.size();
                });
            if (found == jobs.end()) break;
            Complete(*found);
        }
    }

    void Handle(Message& message) {
        auto& state = message.state;
        switch (message.kind) {
        case Message::Kind::Add:
            state->lastServed = dispatchTick;
            jobs.push_back(state);
            if (shutdownSeen) CancelAll();
            else Dispatch();
            return;
        case Message::Kind::Shutdown:
            shutdownSeen = true;
            CancelAll();
            return;
        case Message::Kind::Finish:
            if (!state || state->completed) return;
            --state->running;
            --active;
            ++state->settled;
            if (message.error) {
                state->cancelled = true;
                if (!state->error) state->error = std::move(message.error);
            } else if (message.task == state->job.finalTask && message.publish) {
                state->publish = std::move(message.publish);
            }
            if (!state->cancelled)
                for (auto child : state->children[message.task])
                    --state->remaining[child];
            if (state->cancelled)
                for (uint32_t i = 0; i != state->started.size(); ++i)
                    if (!state->started[i]) {
                        state->started[i] = true;
                        ++state->settled;
                    }
            if (state->settled == state->started.size())
                Complete(state);
            Dispatch();
            return;
        }
    }

    // Queues `message` and handles the lane if free. Completion callbacks
    // made here run inline on graph threads (task finishes, as before);
    // producers outside the graph hand them to the workers instead.
    void Post(Message* message, bool inlineCallbacks) noexcept {
        inbox.Push(message);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        for (;;) {
            if (laneBusy.exchange(true, std::memory_order_acquire)) return;
            try {
                while (Message* batch = inbox.PopAll()) {
                    Message* fifo = nullptr;
                    while (batch) {
                        Message* next = batch->next;
                        batch->next = fifo;
                        fifo = batch;
                        batch = next;
                    }
                    while (fifo) {
                        Message* m = fifo;
                        fifo = fifo->next;
                        Handle(*m);
                        if (m == &shutdownMessage) shutdownMessage.state.reset();
                        else delete m;
                    }
                }
            } catch (...) { std::terminate(); }
            CallbackTask* callbacks = deferred;
            deferred = nullptr;
            laneBusy.store(false, std::memory_order_seq_cst);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            CallbackTask* order = nullptr;  // completion order
            while (callbacks) {
                CallbackTask* next = callbacks->nextCallback;
                callbacks->nextCallback = order;
                order = callbacks;
                callbacks = next;
            }
            while (order) {
                CallbackTask* next = order->nextCallback;
                if (inlineCallbacks) RunCallback(order);
                else scheduler->Post(order);
                order = next;
            }
            if (inbox.Empty()) return;
        }
    }

    static void RunCallback(UsdGenTask* task) noexcept {
        std::unique_ptr<CallbackTask> own(static_cast<CallbackTask*>(task));
        {
            TaskGraphScope scope;
            try { own->callback(std::move(own->publish), own->error); }
            catch (...) {}
        }
        own->callback = {};
        own->error = {};
        own->self->pending.Release();
    }

    void Finish(std::shared_ptr<State> state, uint32_t task,
                Publish publish, std::exception_ptr error) noexcept {
        Message* message = nullptr;
        try {
            message = new Message;
        } catch (...) { std::terminate(); }
        message->kind = Message::Kind::Finish;
        message->state = std::move(state);
        message->task = task;
        message->publish = std::move(publish);
        message->error = std::move(error);
        Post(message, true);
    }

    static void RunReady(UsdGenTask* task) noexcept {
        std::shared_ptr<Impl> self;
        std::shared_ptr<State> state;
        uint32_t index = 0;
        {
            std::unique_ptr<RunTask> own(static_cast<RunTask*>(task));
            self = std::move(own->self);
            state = std::move(own->state);
            index = own->task;
        }
        TaskGraphScope scope;
        struct Once {
            std::atomic<bool> done{false};
            std::atomic<unsigned> flags{0};
            Publish publish;
            std::exception_ptr error;
        };
        try {
            auto once = std::make_shared<Once>();
            auto relay = [self, state, task = index, once]() mutable {
                self->Finish(std::move(state), task,
                             std::move(once->publish), once->error);
            };
            auto finish = [once, relay](Publish publish,
                                        std::exception_ptr error) mutable {
                if (once->done.exchange(true, std::memory_order_acq_rel)) return;
                once->publish = std::move(publish);
                once->error = std::move(error);
                if (once->flags.fetch_or(1, std::memory_order_acq_rel) & 2)
                    relay();
            };
            if (self->closing.load(std::memory_order_acquire) ||
                (state->job.cancellation.epoch &&
                 state->job.cancellation.Superseded()))
                finish({}, std::make_exception_ptr(
                    std::runtime_error("task graph cancelled")));
            else
                state->job.tasks[index].run(state->job.cancellation, finish);
            if (once->flags.fetch_or(2, std::memory_order_acq_rel) & 1)
                relay();
        } catch (...) {
            self->Finish(std::move(state), index, {}, std::current_exception());
        }
    }

    bool Add(std::shared_ptr<State> state) {
        // Paired with Shutdown: either this sees closing, or Shutdown's
        // Drain sees the job.
        pending.Add();
        if (closing.load(std::memory_order_seq_cst)) {
            pending.Release();
            return false;
        }
        Message* message = nullptr;
        try {
            message = new Message;
        } catch (...) {
            pending.Release();
            throw;
        }
        message->kind = Message::Kind::Add;
        message->state = std::move(state);
        Post(message, false);
        return true;
    }

    void Drain() noexcept { pending.Wait(); }

    void Shutdown() noexcept {
        if (shutdownStarted.exchange(true, std::memory_order_acq_rel)) return;
        closing.store(true, std::memory_order_seq_cst);
        Post(&shutdownMessage, false);
        Drain();
        scheduler.reset();
    }
};

namespace {
// Append-only: graphs are process-lifetime, so nodes are never removed.
struct RegistryNode {
    RegistryKey key;
    std::shared_ptr<UsdGenExecutionTaskGraph> graph;
    RegistryNode* next = nullptr;
};
std::atomic<RegistryNode*> registryHead{nullptr};

RegistryNode* FindNode(RegistryNode* from, RegistryNode* until,
                       RegistryKey const& key) {
    for (RegistryNode* node = from; node != until; node = node->next)
        if (node->key == key) return node;
    return nullptr;
}
bool SameLimits(UsdGenExecutionTaskGraph::Limits const& a,
                UsdGenExecutionTaskGraph::Limits const& b) {
    return a.maxJobs == b.maxJobs && a.maxTasks == b.maxTasks &&
        a.maxDependencyEdges == b.maxDependencyEdges &&
        a.maxActiveTasks == b.maxActiveTasks &&
        a.interactiveBurst == b.interactiveBurst &&
        a.backgroundAgingDispatches == b.backgroundAgingDispatches;
}
}

UsdGenExecutionTaskGraph::UsdGenExecutionTaskGraph(std::shared_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
UsdGenExecutionTaskGraph::~UsdGenExecutionTaskGraph() {
    if (!impl_) return;
    if (executingTaskGraph) std::terminate();
    Shutdown();
}

std::shared_ptr<UsdGenExecutionTaskGraph>
UsdGenExecutionTaskGraph::GetOrCreate(
    UsdGenExecutionRuntime& runtime, std::string backend, int device,
    Limits limits) {
    (void)runtime;
    if (backend.empty() || device < 0 || !limits.maxJobs ||
        !limits.maxTasks || !limits.maxActiveTasks ||
        !limits.interactiveBurst) return {};
    RegistryKey key{std::move(backend), device};
    auto matching = [&limits](RegistryNode const* node) {
        return SameLimits(node->graph->impl_->limits, limits) ? node->graph :
            std::shared_ptr<UsdGenExecutionTaskGraph>{};
    };
    RegistryNode* head = registryHead.load(std::memory_order_acquire);
    if (RegistryNode* found = FindNode(head, nullptr, key)) return matching(found);
    auto node = std::make_unique<RegistryNode>();
    node->key = key;
    node->graph = std::shared_ptr<UsdGenExecutionTaskGraph>(
        new UsdGenExecutionTaskGraph(std::make_shared<Impl>(limits)));
    for (;;) {
        node->next = head;
        if (registryHead.compare_exchange_weak(head, node.get(),
                std::memory_order_acq_rel, std::memory_order_acquire))
            return node.release()->graph;
        // Only nodes pushed since our scan can match.
        if (RegistryNode* found = FindNode(head, node->next, key)) {
            // Lost the race: retire the unused candidate's idle workers.
            node->graph->impl_->Shutdown();
            node->graph->impl_.reset();
            return matching(found);
        }
    }
}

bool UsdGenExecutionTaskGraph::Submit(Job job) {
    if (!impl_ || impl_->closing.load(std::memory_order_acquire) ||
        job.tasks.empty() || job.tasks.size() > UINT32_MAX ||
        job.tasks.size() > impl_->limits.maxTasks) return false;
    auto const count = static_cast<uint32_t>(job.tasks.size());
    if (job.requestClass != RequestClass::Interactive &&
        job.requestClass != RequestClass::Background) return false;
    if (job.finalTask == UINT32_MAX) job.finalTask = count - 1;
    if (job.finalTask >= count) return false;
    uint32_t edgeCount = 0;
    for (uint32_t i = 0; i != count; ++i) {
        if (!job.tasks[i].run) return false;
        for (auto dependency : job.tasks[i].dependencies) {
            if (dependency >= count || dependency == i ||
                edgeCount == impl_->limits.maxDependencyEdges) return false;
            ++edgeCount;
        }
    }
    if (edgeCount > impl_->limits.maxDependencyEdges) return false;
    auto jobs = impl_->admittedJobs.load(std::memory_order_relaxed);
    do {
        if (jobs >= impl_->limits.maxJobs) return false;
    } while (!impl_->admittedJobs.compare_exchange_weak(
        jobs, jobs + 1, std::memory_order_acq_rel,
        std::memory_order_relaxed));
    auto tasks = impl_->admittedTasks.load(std::memory_order_relaxed);
    do {
        if (count > impl_->limits.maxTasks - tasks) {
            impl_->admittedJobs.fetch_sub(1, std::memory_order_acq_rel);
            return false;
        }
    } while (!impl_->admittedTasks.compare_exchange_weak(
        tasks, tasks + count, std::memory_order_acq_rel,
        std::memory_order_relaxed));
    auto edges = impl_->admittedEdges.load(std::memory_order_relaxed);
    do {
        if (edgeCount > impl_->limits.maxDependencyEdges - edges) {
            impl_->admittedTasks.fetch_sub(count, std::memory_order_acq_rel);
            impl_->admittedJobs.fetch_sub(1, std::memory_order_acq_rel);
            return false;
        }
    } while (!impl_->admittedEdges.compare_exchange_weak(
        edges, edges + edgeCount, std::memory_order_acq_rel,
        std::memory_order_relaxed));
    struct Reservation {
        Impl* impl;
        uint32_t tasks;
        uint32_t edges;
        bool transferred = false;
        ~Reservation() {
            if (transferred) return;
            impl->admittedEdges.fetch_sub(edges, std::memory_order_acq_rel);
            impl->admittedTasks.fetch_sub(tasks, std::memory_order_acq_rel);
            impl->admittedJobs.fetch_sub(1, std::memory_order_acq_rel);
        }
    } reservation{impl_.get(), count, edgeCount};
    std::vector<uint32_t> indegree(count);
    std::vector<std::vector<uint32_t>> children(count);
    for (uint32_t i = 0; i != count; ++i)
        for (auto dependency : job.tasks[i].dependencies) {
            ++indegree[i];
            children[dependency].push_back(i);
        }
    auto test = indegree;
    std::vector<uint32_t> ready;
    for (uint32_t i = 0; i != count; ++i)
        if (!test[i]) ready.push_back(i);
    uint32_t seen = 0;
    for (size_t position = 0; position != ready.size(); ++position) {
        ++seen;
        for (auto child : children[ready[position]])
            if (!--test[child]) ready.push_back(child);
    }
    if (seen != count) return false;
    std::vector<bool> ancestor(count);
    std::vector<uint32_t> stack{job.finalTask};
    while (!stack.empty()) {
        auto const index = stack.back();
        stack.pop_back();
        if (ancestor[index]) continue;
        ancestor[index] = true;
        for (auto dependency : job.tasks[index].dependencies)
            stack.push_back(dependency);
    }
    for (bool used : ancestor) if (!used) return false;
    auto state = std::make_shared<Impl::State>();
    state->job = std::move(job);
    state->remaining = std::move(indegree);
    state->children = std::move(children);
    state->started.resize(count);
    state->edgeCount = edgeCount;
    if (!impl_->Add(std::move(state))) return false;
    reservation.transferred = true;
    return true;
}

void UsdGenExecutionTaskGraph::Drain() {
    if (usdGenExecutionGraphActive)
        throw std::logic_error(
            "graph work must not synchronously drain task dispatcher");
    if (impl_) impl_->Drain();
}

UsdGenExecutionTaskGraph::AdmissionUsage
UsdGenExecutionTaskGraph::GetAdmissionUsage() const noexcept {
    if (!impl_) return {};
    return {impl_->admittedJobs.load(std::memory_order_acquire),
            impl_->admittedTasks.load(std::memory_order_acquire),
            impl_->admittedEdges.load(std::memory_order_acquire)};
}

void UsdGenExecutionTaskGraph::Shutdown() {
    if (usdGenExecutionGraphActive) std::terminate();
    if (impl_) impl_->Shutdown();
}
}
