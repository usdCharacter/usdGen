// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/exec/taskScheduler.h"

#include "usdGen/exec/epoch.h"
#include "usdGen/exec/executionRuntimeInternal.h"
#include "usdGen/exec/parkingWord.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <thread>
#include <vector>

namespace usdGen {

struct UsdGenTaskScheduler::State {
    static constexpr size_t kRing = 1024;
    struct Cell {
        std::atomic<size_t> seq{0};
        UsdGenTask *task = nullptr;
    };
    alignas(64) std::atomic<size_t> enq{0};
    alignas(64) std::atomic<size_t> deq{0};
    alignas(64) UsdGenWakeWord wake;
    std::atomic<bool> stop{false};
    std::atomic<uint32_t> refs{1};  // owner + one per live worker
    int workers = 0;
    UsdGenTreiberStack<UsdGenTask, &UsdGenTask::taskNext> overflow;
    std::vector<std::thread> threads;
    Cell ring[kRing];

    State() { for (size_t i = 0; i != kRing; ++i) ring[i].seq.store(i, std::memory_order_relaxed); }

    void Unref() noexcept { if (refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this; }

    bool TryEnqueue(UsdGenTask *t) noexcept
    {
        size_t pos = enq.load(std::memory_order_relaxed);
        for (;;) {
            Cell &c = ring[pos & (kRing - 1)];
            size_t const seq = c.seq.load(std::memory_order_acquire);
            intptr_t const dif = intptr_t(seq) - intptr_t(pos);
            if (dif == 0) {
                if (enq.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    c.task = t;
                    c.seq.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (dif < 0) {
                return false;  // full
            } else {
                pos = enq.load(std::memory_order_relaxed);
            }
        }
    }
    UsdGenTask *TryDequeue() noexcept
    {
        size_t pos = deq.load(std::memory_order_relaxed);
        for (;;) {
            Cell &c = ring[pos & (kRing - 1)];
            size_t const seq = c.seq.load(std::memory_order_acquire);
            intptr_t const dif = intptr_t(seq) - intptr_t(pos + 1);
            if (dif == 0) {
                if (deq.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {
                    UsdGenTask *t = c.task;
                    c.seq.store(pos + kRing, std::memory_order_release);
                    return t;
                }
            } else if (dif < 0) {
                return nullptr;  // empty (or a producer mid-publish; it wakes after)
            } else {
                pos = deq.load(std::memory_order_relaxed);
            }
        }
    }
    // The overflow is a stack: a pop takes all of it, keeps the oldest task
    // and hands the rest to the ring in age order (what does not fit goes
    // back), then wakes sleepers, since a worker that found the overflow
    // empty meanwhile may have parked with these tasks queued.
    UsdGenTask *TryPop() noexcept
    {
        if (UsdGenTask *t = TryDequeue()) return t;
        UsdGenTask *chain = overflow.PopAll();  // newest first
        if (!chain) return nullptr;
        UsdGenTask *oldest = nullptr;
        while (chain) {
            UsdGenTask *const n = chain->taskNext;
            chain->taskNext = oldest;
            oldest = chain;
            chain = n;
        }
        UsdGenTask *rest = oldest->taskNext;
        oldest->taskNext = nullptr;
        if (rest) Republish(rest);
        return oldest;
    }
    void Republish(UsdGenTask *rest) noexcept  // oldest first
    {
        bool const several = rest->taskNext != nullptr;
        while (rest) {
            UsdGenTask *const n = rest->taskNext;  // read before it can run
            rest->taskNext = nullptr;
            if (!TryEnqueue(rest)) {
                rest->taskNext = n;
                break;
            }
            rest = n;
        }
        if (rest) {
            UsdGenTask *const last = rest;  // oldest left over: the bottom
            UsdGenTask *head = nullptr;
            while (rest) {
                UsdGenTask *const n = rest->taskNext;
                rest->taskNext = head;
                head = rest;
                rest = n;
            }
            overflow.PushChain(head, last);
        }
        if (several) wake.NotifyAll();
        else wake.NotifyOne();
    }
    void Post(UsdGenTask *t) noexcept
    {
        t->taskNext = nullptr;
        if (!TryEnqueue(t)) overflow.Push(t);
        wake.NotifyOne();
    }
    void Loop() noexcept;
};

namespace {
thread_local UsdGenTaskScheduler::State const *tlsWorkerOf = nullptr;
std::atomic<UsdGenTaskScheduler *> processScheduler{nullptr};
}

void UsdGenTaskScheduler::State::Loop() noexcept
{
    tlsWorkerOf = this;
    for (;;) {
        if (UsdGenTask *t = TryPop()) {
            t->taskRun(t);
            continue;
        }
        // Idle: free what earlier tasks displaced (cells, snapshots) before
        // sleeping, so retired values do not linger while the engine rests.
        // Never blocks; a no-op with nothing pending or a reader pinned.
        if (UsdGenEpochDrain()) continue;
        uint32_t const token = wake.Announce();
        if (UsdGenTask *t = TryPop()) {
            wake.Cancel();
            t->taskRun(t);
            continue;
        }
        if (stop.load(std::memory_order_acquire)) {
            wake.Cancel();
            break;
        }
        wake.Park(token);
    }
    tlsWorkerOf = nullptr;
    Unref();
}

namespace {
void StopState(UsdGenTaskScheduler::State *s) noexcept
{
    s->stop.store(true, std::memory_order_seq_cst);
    s->wake.WakeAll();
    // Process detach: the workers are gone or must not be waited for.
    bool const detach = UsdGenProcessShutdownInProgress();
    auto const self = std::this_thread::get_id();
    for (auto &t : s->threads) {
        if (!t.joinable()) continue;
        if (detach || t.get_id() == self) t.detach();
        else t.join();
    }
    s->Unref();
}
}  // namespace

UsdGenTaskScheduler::UsdGenTaskScheduler(int workers) : _state(new State)
{
    State *s = _state;
    s->workers = std::max(1, workers);
    try {
        s->threads.reserve(size_t(s->workers));
        for (int i = 0; i != s->workers; ++i) {
            s->refs.fetch_add(1, std::memory_order_relaxed);
            try {
                s->threads.emplace_back([s] { s->Loop(); });
            } catch (...) {
                s->refs.fetch_sub(1, std::memory_order_relaxed);
                throw;
            }
        }
    } catch (...) {
        _state = nullptr;
        StopState(s);
        throw;
    }
}

UsdGenTaskScheduler::~UsdGenTaskScheduler()
{
    if (State *s = _state) {
        _state = nullptr;
        StopState(s);
    }
}

int UsdGenTaskScheduler::Workers() const noexcept { return _state ? _state->workers : 0; }

void UsdGenTaskScheduler::Post(UsdGenTask *task) noexcept { _state->Post(task); }

bool UsdGenTaskScheduler::OnWorker() const noexcept { return _state && tlsWorkerOf == _state; }

UsdGenTaskScheduler &UsdGenTaskScheduler::Process()
{
    if (UsdGenTaskScheduler *p = processScheduler.load(std::memory_order_acquire)) return *p;
    unsigned const hw = std::thread::hardware_concurrency();
    auto mine = std::make_unique<UsdGenTaskScheduler>(int(std::max(2u, hw ? hw : 8u)));
    UsdGenTaskScheduler *expected = nullptr;
    if (processScheduler.compare_exchange_strong(expected, mine.get(), std::memory_order_acq_rel,
                                                 std::memory_order_acquire))
        return *mine.release();
    return *expected;  // lost the race: `mine` joins its idle workers
}

void UsdGenPendingCount::Release(uint32_t n) noexcept
{
    uint32_t const old = _word.fetch_sub(n, std::memory_order_seq_cst);
    if (old == (kWaiter | n)) UsdGenUnparkAll(_word);
}

void UsdGenPendingCount::Wait() noexcept
{
    for (;;) {
        uint32_t v = _word.load(std::memory_order_seq_cst);
        if (!(v & ~kWaiter)) {
            if (v == kWaiter) {
                uint32_t expected = kWaiter;
                _word.compare_exchange_strong(expected, 0, std::memory_order_relaxed);
            }
            return;
        }
        if (!(v & kWaiter) &&
            !_word.compare_exchange_weak(v, v | kWaiter, std::memory_order_seq_cst))
            continue;
        UsdGenParkWhileEqual(_word, v | kWaiter);
    }
}

namespace {

// One fork/join region. Helpers join by incrementing `state` unless CLOSED
// is set; the caller sets CLOSED once the range is exhausted and parks until
// the joined helpers left. The block is freed by its last reference.
struct ForRegion {
    static constexpr uint32_t kClosed = 0x80000000u;
    struct Helper : UsdGenTask {
        ForRegion *region = nullptr;
    };
    std::atomic<uint32_t> refs{0};
    std::atomic<uint32_t> state{0};
    std::atomic<size_t> next{0};
    std::atomic<bool> failed{false};
    std::exception_ptr error;
    void (*body)(size_t, void *) = nullptr;
    void *payload = nullptr;
    size_t count = 0;
    size_t grain = 1;
    std::unique_ptr<Helper[]> helpers;

    void Unref() noexcept { if (refs.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this; }

    void Work() noexcept
    {
        for (;;) {
            if (failed.load(std::memory_order_relaxed)) return;
            size_t const begin = next.fetch_add(grain, std::memory_order_relaxed);
            if (begin >= count) return;
            size_t const end = std::min(count, begin + grain);
            try {
                for (size_t i = begin; i != end; ++i) body(i, payload);
            } catch (...) {
                if (!failed.exchange(true, std::memory_order_acq_rel)) error = std::current_exception();
                return;
            }
        }
    }

    static void RunHelper(UsdGenTask *t) noexcept
    {
        ForRegion *r = static_cast<Helper *>(t)->region;
        uint32_t s = r->state.load(std::memory_order_acquire);
        bool joined = false;
        while (!(s & kClosed)) {
            if (r->state.compare_exchange_weak(s, s + 1, std::memory_order_acq_rel,
                                               std::memory_order_acquire)) {
                joined = true;
                break;
            }
        }
        if (joined) {
            r->Work();
            if (r->state.fetch_sub(1, std::memory_order_acq_rel) == (kClosed | 1))
                UsdGenUnparkAll(r->state);
        }
        r->Unref();
    }
};

}  // namespace

int UsdGenParallelForConcurrency()
{
    return UsdGenTaskScheduler::Process().Workers() + 1;
}

void UsdGenParallelFor(size_t count, void (*body)(size_t, void *), void *payload, size_t maxWorkers)
{
    if (!count) return;
    UsdGenTaskScheduler &scheduler = UsdGenTaskScheduler::Process();
    size_t threads = size_t(scheduler.Workers()) + 1;
    if (maxWorkers) threads = std::min(threads, maxWorkers);
    threads = std::min(threads, count);
    if (threads <= 1) {
        for (size_t i = 0; i != count; ++i) body(i, payload);
        return;
    }
    size_t const helpers = threads - 1;
    auto *r = new ForRegion;
    r->helpers.reset(new ForRegion::Helper[helpers]);
    r->body = body;
    r->payload = payload;
    r->count = count;
    // ~4 claims per thread: small enough to balance, big enough for locality.
    r->grain = std::max<size_t>(1, count / (threads * 4));
    r->refs.store(uint32_t(helpers + 1), std::memory_order_relaxed);
    for (size_t i = 0; i != helpers; ++i) {
        r->helpers[i].taskRun = &ForRegion::RunHelper;
        r->helpers[i].region = r;
        scheduler.Post(&r->helpers[i]);
    }
    r->Work();
    uint32_t s = r->state.fetch_or(ForRegion::kClosed, std::memory_order_acq_rel) | ForRegion::kClosed;
    while (s != ForRegion::kClosed) {
        UsdGenParkWhileEqual(r->state, s);
        s = r->state.load(std::memory_order_acquire);
    }
    std::exception_ptr error = r->failed.load(std::memory_order_acquire) ? r->error : nullptr;
    r->Unref();
    if (error) std::rethrow_exception(error);
}

void UsdGenParallelForBlocks(void *, size_t count,
                             void (*body)(void *user, size_t begin, size_t end), void *user)
{
    struct Blocks {
        void (*body)(void *, size_t, size_t);
        void *user;
    } blocks{body, user};
    UsdGenParallelFor(count, [](size_t i, void *p) {
        Blocks const &b = *static_cast<Blocks const *>(p);
        b.body(b.user, i, i + 1);
    }, &blocks);
}

}  // namespace usdGen
