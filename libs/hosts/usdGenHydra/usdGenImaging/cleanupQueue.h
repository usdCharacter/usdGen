// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen imaging — cleanup actions run off the posting thread.
//
// UsdGenCleanupQueue runs each posted action as its own task on the process
// task scheduler (usdGen/taskScheduler.h), never inline: a final-reference
// release or a pipeline callback that posts never runs the action itself,
// and actions run concurrently, as the flow-graph node with unlimited
// concurrency did. Wait parks until every action posted before it finished
// and rethrows the first exception an action threw since the last Wait.
//
// Queues are process-lifetime and leaked (UsdGenLazyLeaked): the destructor
// does not wait, because at DLL detach no worker remains to run a task.
#ifndef USDGEN_IMAGING_CLEANUP_QUEUE_H
#define USDGEN_IMAGING_CLEANUP_QUEUE_H

#include "usdGen/api/taskScheduler.h"

#include <atomic>
#include <exception>
#include <functional>
#include <new>
#include <utility>

namespace usdGenImaging {

class UsdGenCleanupQueue {
public:
    UsdGenCleanupQueue() = default;
    UsdGenCleanupQueue(UsdGenCleanupQueue const &) = delete;
    UsdGenCleanupQueue &operator=(UsdGenCleanupQueue const &) = delete;

    /// Queues `action`; throws (queuing nothing) only if allocation fails.
    void Post(std::function<void()> action)
    {
        auto *task = new Task(std::move(action), this);
        _pending.Add();
        usdGen::UsdGenTaskScheduler::Process().Post(task);
    }
    /// Parks until no posted action is outstanding.
    void Wait()
    {
        _pending.Wait();
        if (std::exception_ptr *error = _error.exchange(nullptr, std::memory_order_acq_rel)) {
            std::exception_ptr rethrow = std::move(*error);
            delete error;
            std::rethrow_exception(rethrow);
        }
    }
    uint32_t Outstanding() const noexcept { return _pending.Count(); }

private:
    struct Task : usdGen::UsdGenTask {
        Task(std::function<void()> a, UsdGenCleanupQueue *q)
            : usdGen::UsdGenTask(&Run), action(std::move(a)), queue(q) {}
        static void Run(usdGen::UsdGenTask *t) noexcept
        {
            auto *self = static_cast<Task *>(t);
            UsdGenCleanupQueue *const queue = self->queue;
            try {
                self->action();
            } catch (...) {
                queue->_Fail(std::current_exception());
            }
            delete self;
            queue->_pending.Release();
        }
        std::function<void()> action;
        UsdGenCleanupQueue *queue;
    };
    void _Fail(std::exception_ptr e) noexcept
    {
        auto *mine = new (std::nothrow) std::exception_ptr(std::move(e));
        std::exception_ptr *expected = nullptr;
        if (mine && !_error.compare_exchange_strong(expected, mine, std::memory_order_acq_rel))
            delete mine;  // an earlier failure is already recorded
    }
    usdGen::UsdGenPendingCount _pending;
    std::atomic<std::exception_ptr *> _error{nullptr};
};

}  // namespace usdGenImaging

#endif  // USDGEN_IMAGING_CLEANUP_QUEUE_H
