#include "usdGen/executionPipeline.h"
#include "usdGen/executionRuntimeInternal.h"

#include "usdGen/epoch.h"
#include "usdGen/parkingWord.h"
#include "usdGen/taskScheduler.h"

#include <optional>
#include <stdexcept>
#include <utility>

namespace usdGen {
thread_local bool usdGenExecutionGraphActive = false;
namespace {
thread_local bool executingPipeline = false;
thread_local void const* executingOwner = nullptr;
struct ExecutionScope {
    bool previous = executingPipeline;
    bool previousActive = usdGenExecutionGraphActive;
    void const* previousOwner = executingOwner;
    explicit ExecutionScope(void const* owner = nullptr) {
        executingPipeline = true; usdGenExecutionGraphActive = true; executingOwner = owner;
    }
    ~ExecutionScope() { executingPipeline = previous; usdGenExecutionGraphActive = previousActive; executingOwner = previousOwner; }
};

// One latch per external reply, parked on its own word. The signalling
// closure owns the state, so a late signal never touches a returned waiter.
struct AwaitState {
    std::atomic<uint32_t> signalled{0};
    void Signal() {
        if (!signalled.exchange(1, std::memory_order_acq_rel)) UsdGenUnparkAll(signalled);
    }
    void Wait() {
        while (!signalled.load(std::memory_order_acquire)) UsdGenParkWhileEqual(signalled, 0);
    }
};

struct LatestCommand : UsdGenRefCounted {
    explicit LatestCommand(std::function<void()> f) : fn(std::move(f)) {}
    std::function<void()> fn;
};
}

UsdGenExecutionRuntime::UsdGenExecutionRuntime(int concurrency) {
    if (concurrency < 1) throw std::invalid_argument("execution concurrency must be positive");
    impl_ = std::make_shared<Impl>(concurrency);
}
UsdGenExecutionRuntime::~UsdGenExecutionRuntime() = default;

struct UsdGenExecutionPipeline::CommandTicket::State {
    std::shared_ptr<std::atomic<uint64_t>> outstanding;
    std::shared_ptr<void> identity;
    std::atomic<bool> released{false};
    void Release() noexcept {
        if (!released.exchange(true, std::memory_order_acq_rel))
            outstanding->fetch_sub(1, std::memory_order_acq_rel);
    }
    ~State() { Release(); }
};

struct UsdGenExecutionPipeline::CommandMailbox::State {
    std::shared_ptr<std::atomic<uint64_t>> outstanding;
    std::shared_ptr<void> identity;
    std::atomic<bool> scheduled{false};
    std::atomic<bool> broken{false};
    UsdGenNewestWins<LatestCommand> latest;
    ~State() { outstanding->fetch_sub(1, std::memory_order_acq_rel); }
};

UsdGenExecutionPipeline::CommandTicket::CommandTicket() noexcept = default;
UsdGenExecutionPipeline::CommandTicket::CommandTicket(
    std::shared_ptr<State> state) noexcept : state_(std::move(state)) {}
UsdGenExecutionPipeline::CommandTicket::~CommandTicket() = default;
UsdGenExecutionPipeline::CommandTicket::CommandTicket(CommandTicket&&) noexcept = default;
UsdGenExecutionPipeline::CommandTicket&
UsdGenExecutionPipeline::CommandTicket::operator=(CommandTicket&&) noexcept = default;
UsdGenExecutionPipeline::CommandTicket::operator bool() const noexcept {
    return static_cast<bool>(state_);
}
UsdGenExecutionPipeline::CommandMailbox::CommandMailbox() noexcept = default;
UsdGenExecutionPipeline::CommandMailbox::CommandMailbox(
    std::shared_ptr<State> state) noexcept : state_(std::move(state)) {}
UsdGenExecutionPipeline::CommandMailbox::~CommandMailbox() = default;
UsdGenExecutionPipeline::CommandMailbox::CommandMailbox(CommandMailbox const&) noexcept = default;
UsdGenExecutionPipeline::CommandMailbox&
UsdGenExecutionPipeline::CommandMailbox::operator=(CommandMailbox const&) noexcept = default;
UsdGenExecutionPipeline::CommandMailbox::CommandMailbox(CommandMailbox&&) noexcept = default;
UsdGenExecutionPipeline::CommandMailbox&
UsdGenExecutionPipeline::CommandMailbox::operator=(CommandMailbox&&) noexcept = default;
UsdGenExecutionPipeline::CommandMailbox::operator bool() const noexcept {
    return static_cast<bool>(state_);
}

struct UsdGenExecutionPipeline::Impl {
    struct Message {
        enum class Kind { Request, Cancel, Finished, Command, Mailbox } kind = Kind::Request;
        uint64_t epoch = 0;
        Work work;
        AsyncWork asyncWork;
        Publish publish;
        Completion completion;
        std::exception_ptr error;
        bool requestCredit = false;
        std::shared_ptr<CommandTicket::State> commandTicket;
        std::shared_ptr<CommandMailbox::State> mailbox;
    };
    // A serial lane: producers push on a Treiber stack and the one that sets
    // `scheduled` posts the lane to the runtime scheduler. A run handles the
    // whole batch in FIFO order, so one lane never runs on two threads.
    struct MessageNode {
        MessageNode* next = nullptr;
        Message message;
    };
    struct Lane : UsdGenTask {
        Impl* impl = nullptr;
        bool isOwner = false;
        UsdGenTreiberStack<MessageNode, &MessageNode::next> inbox;
        std::atomic<bool> scheduled{false};
    };
    std::shared_ptr<UsdGenExecutionRuntime::Impl> runtime;
    std::shared_ptr<std::atomic<uint64_t>> accepted =
        std::make_shared<std::atomic<uint64_t>>(0);
    std::atomic<uint64_t> requested{0}, callbackFailures{0};
    std::atomic<uint64_t> outstandingRequests{0};
    std::shared_ptr<std::atomic<uint64_t>> outstandingCommands =
        std::make_shared<std::atomic<uint64_t>>(0);
    const uint64_t requestCapacity;
    const uint64_t commandCapacity;
    std::shared_ptr<void> commandIdentity = std::make_shared<char>();
    std::shared_ptr<std::atomic<bool>> closing = std::make_shared<std::atomic<bool>>(false);
    // Queued lane messages plus external replies still held by async work.
    // Drain and Shutdown park until it reaches zero.
    UsdGenPendingCount pending;
    Lane owner;
    Lane worker;
    // Owner-only state. Never put a second request on the worker lane while
    // workRunning is true.
    bool workRunning = false;
    std::optional<Message> pendingWork;

    explicit Impl(std::shared_ptr<UsdGenExecutionRuntime::Impl> runtime_,
                  uint64_t requestCapacity_, uint64_t commandCapacity_)
        : runtime(std::move(runtime_)), requestCapacity(requestCapacity_),
          commandCapacity(commandCapacity_) {
        owner.taskRun = &RunLane;
        owner.impl = this;
        owner.isOwner = true;
        worker.taskRun = &RunLane;
        worker.impl = this;
    }

    // Allocates before moving, so a throw leaves `message` intact.
    void Put(Lane& lane, Message& message) {
        auto* node = new MessageNode;
        node->message = std::move(message);
        pending.Add();
        lane.inbox.Push(node);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (!lane.scheduled.exchange(true, std::memory_order_acq_rel))
            runtime->scheduler.Post(&lane);
    }
    void PutOwner(Message message) { Put(owner, message); }

    static void RunLane(UsdGenTask* task) noexcept {
        auto& lane = *static_cast<Lane*>(task);
        Impl& impl = *lane.impl;
        MessageNode* batch = nullptr;
        for (MessageNode* n = lane.inbox.PopAll(); n;) {
            MessageNode* next = n->next;
            n->next = batch;
            batch = n;
            n = next;
        }
        uint32_t handled = 0;
        while (batch) {
            std::unique_ptr<MessageNode> node(batch);
            batch = batch->next;
            ++handled;
            try {
                if (lane.isOwner) {
                    ExecutionScope scope(&impl);
                    impl.Accept(std::move(node->message));
                } else {
                    ExecutionScope scope;
                    impl.RunWork(std::move(node->message));
                }
            } catch (...) {
                impl.callbackFailures.fetch_add(1, std::memory_order_relaxed);
            }
        }
        lane.scheduled.store(false, std::memory_order_seq_cst);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        if (!lane.inbox.Empty() && !lane.scheduled.exchange(true, std::memory_order_acq_rel))
            impl.runtime->scheduler.Post(&lane);
        // Last touch: a drained waiter may free `impl` once this lands.
        impl.pending.Release(handled);
    }

    void RunWork(Message message) {
        Cancellation cancel;
        cancel.epoch = message.epoch;
        cancel.accepted = accepted;
        cancel.closing = closing;
        if (message.asyncWork && !cancel.Superseded()) {
            // The pending hold lasts until an external backend reply has
            // made the terminal owner message. `settled` lives in the
            // callback, so a duplicate/late reply never dereferences this
            // Impl after its first call.
            struct Reply {
                std::atomic<bool> settled{false};
                std::atomic<unsigned> flags{0}; // callback=1, Run returned=2
                Publish publish; std::exception_ptr error;
                Message message;
                explicit Reply(Message&& value) : message(std::move(value)) {}
            };
            // Move the callable out before moving the message into the
            // reply closure.  The reply owns retained request state;
            // invoking a moved-from std::function would throw.
            auto reply = std::make_shared<Reply>(std::move(message));
            auto asyncWork = std::move(reply->message.asyncWork);
            pending.Add();
            auto relay = [this, reply]() mutable {
                reply->message.publish = std::move(reply->publish);
                reply->message.error = std::move(reply->error);
                reply->message.asyncWork = {}; reply->message.work = {}; reply->message.kind = Message::Kind::Finished;
                // Empty the late-callback Reply before delivering the
                // terminal; otherwise it retains user captures.
                Message terminal=std::move(reply->message);
                reply->message = {};
                try { PutOwner(std::move(terminal)); }
                catch (...) { std::terminate(); }
                pending.Release();
            };
            auto done = [reply, relay](Publish publish, std::exception_ptr error) mutable {
                if (reply->settled.exchange(true, std::memory_order_acq_rel)) return;
                reply->publish=std::move(publish); reply->error=std::move(error);
                if (reply->flags.fetch_or(1, std::memory_order_acq_rel) & 2) relay();
            };
            try { asyncWork(cancel, done); }
            catch (...) { done({}, std::current_exception()); }
            if (reply->flags.fetch_or(2, std::memory_order_acq_rel) & 1) relay();
            return;
        }
        if (!cancel.Superseded()) {
            try { message.publish = message.work(cancel); }
            catch (...) { message.error = std::current_exception(); }
        }
        message.work = {};
        message.kind = Message::Kind::Finished;
        PutOwner(std::move(message));
    }

    uint64_t Reserve() {
        auto value = requested.load(std::memory_order_relaxed);
        do {
            if (value == UINT64_MAX || closing->load(std::memory_order_acquire)) return 0;
        } while (!requested.compare_exchange_weak(value, value + 1,
                     std::memory_order_relaxed, std::memory_order_relaxed));
        return value + 1;
    }
    bool AcquireRequestCredit() {
        auto value = outstandingRequests.load(std::memory_order_relaxed);
        do {
            if (value >= requestCapacity) return false;
        } while (!outstandingRequests.compare_exchange_weak(
            value, value + 1, std::memory_order_acq_rel, std::memory_order_relaxed));
        return true;
    }
    bool AcquireCommandCredit() {
        auto value = outstandingCommands->load(std::memory_order_relaxed);
        do {
            if (value >= commandCapacity || closing->load(std::memory_order_acquire)) return false;
        } while (!outstandingCommands->compare_exchange_weak(
            value, value + 1, std::memory_order_acq_rel, std::memory_order_relaxed));
        return true;
    }
    std::shared_ptr<CommandTicket::State> ReserveCommandTicket() {
        if (!AcquireCommandCredit()) return {};
        try {
            auto ticket = std::make_shared<CommandTicket::State>();
            ticket->outstanding = outstandingCommands;
            ticket->identity = commandIdentity;
            return ticket;
        } catch (...) {
            outstandingCommands->fetch_sub(1, std::memory_order_acq_rel);
            throw;
        }
    }
    std::shared_ptr<CommandMailbox::State> ReserveCommandMailbox() {
        if (!AcquireCommandCredit()) return {};
        try {
            auto mailbox = std::make_shared<CommandMailbox::State>();
            mailbox->outstanding = outstandingCommands;
            mailbox->identity = commandIdentity;
            return mailbox;
        } catch (...) {
            outstandingCommands->fetch_sub(1, std::memory_order_acq_rel);
            throw;
        }
    }
    void ReleaseRequestCredit(Message const& message) noexcept {
        if (message.requestCredit)
            outstandingRequests.fetch_sub(1, std::memory_order_acq_rel);
    }
    void Notify(Message const& message, Outcome outcome) noexcept {
        if (!message.completion) return;
        try { message.completion(message.epoch, outcome, message.error); }
        catch (...) { callbackFailures.fetch_add(1, std::memory_order_relaxed); }
    }
    // A request owns one ingress credit until its single terminal callback.
    // Release before Notify: completion reentry can submit even at capacity.
    void Retire(Message message, Outcome outcome) noexcept {
        // Do not retain superseded work/publication captures through a callback
        // which may recursively submit more work.
        message.work = {};
        message.asyncWork = {};
        message.publish = {};
        if (message.commandTicket) message.commandTicket->Release();
        ReleaseRequestCredit(message);
        Notify(message, outcome);
    }
    void Start(Message message) {
        workRunning = true;
        // Used only with no other worker item queued by this owner.
        try { Put(worker, message); return; }
        catch (...) {}
        workRunning = false;
        Retire(std::move(message), Outcome::Superseded);
    }
    void Finish(Message message) {
        if (message.epoch != accepted->load(std::memory_order_acquire)) {
            Retire(std::move(message), Outcome::Superseded);
            return;
        }
        if (!message.error && !message.publish)
            message.error = std::make_exception_ptr(
                std::runtime_error("work returned no publication"));
        if (!message.error && message.publish) {
            try { message.publish(); }
            catch (...) { message.error = std::current_exception(); }
        }
        const auto outcome = message.error ? Outcome::Failed : Outcome::Published;
        Retire(std::move(message), outcome);
    }
    void Accept(Message message) {
        if (closing->load(std::memory_order_acquire)) {
            // A Finished message is the only path that can leave an owner-side
            // pending payload behind during shutdown.
            if (message.kind == Message::Kind::Finished) {
                workRunning = false;
                if (pendingWork) {
                    auto pending = std::move(*pendingWork);
                    pendingWork.reset();
                    Retire(std::move(pending), Outcome::Superseded);
                }
            }
            if (message.kind == Message::Kind::Mailbox && message.mailbox) {
                message.mailbox->latest.Take();
                message.mailbox->scheduled.store(false, std::memory_order_release);
            }
            Retire(std::move(message), Outcome::Superseded);
            return;
        }
        if (message.kind == Message::Kind::Command) {
            // Do not retain the admission ticket through user code: an owner
            // callback may reserve/post its successor even at capacity one.
            auto command = std::move(message.publish);
            if (message.commandTicket) message.commandTicket->Release();
            message.commandTicket.reset();
            try { command(); }
            catch (...) { callbackFailures.fetch_add(1, std::memory_order_relaxed); }
            return;
        }
        if (message.kind == Message::Kind::Mailbox) {
            auto mailbox = std::move(message.mailbox);
            auto command = mailbox->latest.Take();
            mailbox->scheduled.store(false, std::memory_order_seq_cst);
            if (command) {
                try { command->fn(); }
                catch (...) { callbackFailures.fetch_add(1, std::memory_order_relaxed); }
            }
            // A producer may have replaced latest while this owner frame ran.
            // Schedule one successor, never poll or retry in a loop. Store
            // scheduled, then load latest; the producer stores latest, then
            // loads scheduled: both sides fence, or each can miss the other
            // and leave the newest command behind.
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (mailbox->latest.HasValue()) {
                bool expected = false;
                if (mailbox->scheduled.compare_exchange_strong(expected, true, std::memory_order_acq_rel)) {
                    Message successor;
                    successor.kind = Message::Kind::Mailbox;
                    successor.mailbox = mailbox;
                    try {
                        PutOwner(std::move(successor));
                    } catch (...) {
                        mailbox->broken.store(true, std::memory_order_release);
                        mailbox->latest.Take();
                        mailbox->scheduled.store(false, std::memory_order_release);
                        throw;
                    }
                }
            }
            return;
        }
        if (message.kind == Message::Kind::Finished) {
            // Keep the work lane occupied through the accepted publication and
            // terminal callback.  A publication owns the final mutable
            // handoff from its worker (for example a SessionCooker's cache
            // admission and snapshot fence); starting pending work first
            // would let that next worker mutate the same cooker concurrently.
            // Reentrant Submit/CancelPending remains safe: while Finish runs
            // it replaces/cancels pendingWork rather than starting a second
            // worker, and the latest survivor starts immediately below.
            Finish(std::move(message));
            workRunning = false;
            if (pendingWork) {
                auto pending = std::move(*pendingWork);
                pendingWork.reset();
                Start(std::move(pending));
            }
            return;
        }
        const auto current = accepted->load(std::memory_order_relaxed);
        if (message.epoch <= current) {
            Retire(std::move(message), Outcome::Superseded);
            return;
        }
        accepted->store(message.epoch, std::memory_order_release);
        if (message.kind == Message::Kind::Cancel) {
            if (message.commandTicket) message.commandTicket->Release();
            if (pendingWork) {
                auto pending = std::move(*pendingWork);
                pendingWork.reset();
                Retire(std::move(pending), Outcome::Superseded);
            }
            return;
        }
        if (!workRunning) {
            Start(std::move(message));
            return;
        }
        // Install the replacement before notifying the eviction. Its callback
        // can reenter Accept and replace/cancel this new pending request.
        std::optional<Message> evicted;
        if (pendingWork) evicted.emplace(std::move(*pendingWork));
        pendingWork.emplace(std::move(message));
        if (evicted) Retire(std::move(*evicted), Outcome::Superseded);
    }
};

bool UsdGenExecutionPipeline::Cancellation::Superseded() const noexcept {
    // A default token is the backend-neutral task-graph's uncancelled token.
    if (epoch == 0) return false;
    return !accepted || !closing || closing->load(std::memory_order_acquire) ||
        accepted->load(std::memory_order_acquire) != epoch;
}
UsdGenExecutionPipeline::UsdGenExecutionPipeline(UsdGenExecutionRuntime& runtime,
                                                   uint64_t requestCapacity,
                                                   uint64_t commandCapacity) {
    if (!requestCapacity || !commandCapacity)
        throw std::invalid_argument("execution request and command capacities must be positive");
    impl_ = std::make_unique<Impl>(runtime.impl_, requestCapacity, commandCapacity);
}
UsdGenExecutionPipeline::~UsdGenExecutionPipeline() {
    if (usdGenExecutionGraphActive) std::terminate();
    Shutdown();
}
uint64_t UsdGenExecutionPipeline::Submit(Work work, Completion completion) {
    if (!work) return 0;
    auto epoch = impl_->Reserve();
    if (!epoch) return 0;
    if (!impl_->AcquireRequestCredit()) return 0;
    Impl::Message message;
    message.epoch = epoch;
    message.work = std::move(work);
    message.completion = std::move(completion);
    message.requestCredit = true;
    if (executingOwner == impl_.get()) {
        impl_->Accept(std::move(message));
        return epoch;
    }
    try {
        impl_->PutOwner(std::move(message));
    } catch (...) {
        impl_->outstandingRequests.fetch_sub(1, std::memory_order_acq_rel);
        throw;
    }
    return epoch;
}
uint64_t UsdGenExecutionPipeline::SubmitAsync(AsyncWork work, Completion completion) {
    if (!work) return 0;
    auto epoch = impl_->Reserve();
    if (!epoch || !impl_->AcquireRequestCredit()) return 0;
    Impl::Message message;
    message.epoch = epoch;
    message.asyncWork = std::move(work);
    message.completion = std::move(completion);
    message.requestCredit = true;
    if (executingOwner == impl_.get()) { impl_->Accept(std::move(message)); return epoch; }
    try {
        impl_->PutOwner(std::move(message));
    } catch (...) {
        impl_->outstandingRequests.fetch_sub(1, std::memory_order_acq_rel);
        throw;
    }
    return epoch;
}
uint64_t UsdGenExecutionPipeline::CancelPending() {
    auto epoch = impl_->Reserve();
    if (!epoch) return 0;
    Impl::Message message;
    message.kind = Impl::Message::Kind::Cancel;
    message.epoch = epoch;
    if (executingOwner == impl_.get()) {
        impl_->Accept(std::move(message));
        return epoch;
    }
    auto ticket = impl_->ReserveCommandTicket();
    if (!ticket) return 0;
    message.commandTicket = std::move(ticket);
    impl_->PutOwner(std::move(message));
    return epoch;
}
bool UsdGenExecutionPipeline::PostCommand(std::function<void()> command,
                                         std::function<void()> onCancel) {
    auto ticket = ReserveCommandTicket();
    return PostCommand(std::move(ticket), std::move(command), std::move(onCancel));
}
UsdGenExecutionPipeline::CommandTicket UsdGenExecutionPipeline::ReserveCommandTicket() {
    return CommandTicket(impl_->ReserveCommandTicket());
}
bool UsdGenExecutionPipeline::PostCommand(CommandTicket&& ticket,
                                          std::function<void()> command,
                                          std::function<void()> onCancel) {
    if (!command || !ticket.state_ ||
        ticket.state_->identity != impl_->commandIdentity ||
        impl_->closing->load(std::memory_order_acquire)) return false;
    Impl::Message message;
    message.kind = Impl::Message::Kind::Command;
    message.publish = std::move(command);
    message.commandTicket = std::move(ticket.state_);
    if (onCancel) message.completion = [onCancel=std::move(onCancel)](
        uint64_t, Outcome outcome, std::exception_ptr) {
        if (outcome == Outcome::Superseded) onCancel();
    };
    impl_->PutOwner(std::move(message));
    return true;
}
UsdGenExecutionPipeline::CommandMailbox UsdGenExecutionPipeline::ReserveCommandMailbox() {
    return CommandMailbox(impl_->ReserveCommandMailbox());
}
bool UsdGenExecutionPipeline::PostLatestCommand(CommandMailbox const& mailbox,
                                                 std::function<void()> command) {
    if (!command || !mailbox.state_ ||
        mailbox.state_->identity != impl_->commandIdentity ||
        impl_->closing->load(std::memory_order_acquire) ||
        mailbox.state_->broken.load(std::memory_order_acquire)) return false;
    mailbox.state_->latest.Publish(UsdGenMakeRef<LatestCommand>(std::move(command)));
    // Pairs with the owner's fence between clearing scheduled and its
    // recheck of latest (store-load on both sides).
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (mailbox.state_->broken.load(std::memory_order_acquire)) {
        mailbox.state_->latest.Take();
        return false;
    }
    bool expected = false;
    if (!mailbox.state_->scheduled.compare_exchange_strong(expected, true, std::memory_order_acq_rel))
        return true;
    Impl::Message message;
    message.kind = Impl::Message::Kind::Mailbox;
    message.mailbox = mailbox.state_;
    try {
        impl_->PutOwner(std::move(message));
    } catch (...) {
        mailbox.state_->broken.store(true, std::memory_order_release);
        mailbox.state_->latest.Take();
        mailbox.state_->scheduled.store(false, std::memory_order_release);
        throw;
    }
    return true;
}
bool UsdGenExecutionPipeline::CommandMailboxBroken(CommandMailbox const& mailbox) const noexcept {
    return !mailbox.state_ || mailbox.state_->identity != impl_->commandIdentity ||
        mailbox.state_->broken.load(std::memory_order_acquire);
}
bool UsdGenExecutionPipeline::IsExecuting() noexcept { return usdGenExecutionGraphActive; }
bool UsdGenExecutionPipeline::IsExecutingOwner() const noexcept { return executingOwner == impl_.get(); }

void UsdGenExecutionPipeline::Await(std::function<void(std::function<void()>)> dispatch) {
    if (usdGenExecutionGraphActive) throw std::logic_error("graph work must not synchronously await a reply");
    if (!dispatch) throw std::invalid_argument("missing reply dispatcher");
    // The calling thread parks on its own latch; the runtime's workers make
    // progress independently, so no worker slot is needed for the waiter.
    auto state = std::make_shared<AwaitState>();
    std::exception_ptr dispatchError;
    try { dispatch([state] { state->Signal(); }); }
    catch (...) { dispatchError = std::current_exception(); state->Signal(); }
    state->Wait();
    if (dispatchError) std::rethrow_exception(dispatchError);
}

void UsdGenExecutionPipeline::InvokeOwner(std::function<void()> command) {
    if (!command) throw std::invalid_argument("missing owner command");
    struct Reply { std::exception_ptr error; };
    auto reply = std::make_shared<Reply>();
    Await([&, reply, command=std::move(command)](std::function<void()> done) mutable {
        if (!PostCommand([reply, command=std::move(command), done] {
                try { command(); }
                catch (...) { reply->error = std::current_exception(); }
                done();
            }, [reply, done] {
                reply->error = std::make_exception_ptr(std::runtime_error("pipeline closed before command execution"));
                done();
            })) throw std::runtime_error("pipeline rejected owner command");
    });
    if (reply->error) std::rethrow_exception(reply->error);
}
uint64_t UsdGenExecutionPipeline::AcceptedEpoch() const noexcept {
    return impl_->accepted->load(std::memory_order_acquire);
}
uint64_t UsdGenExecutionPipeline::CallbackFailures() const noexcept {
    return impl_->callbackFailures.load(std::memory_order_acquire);
}
uint64_t UsdGenExecutionPipeline::OutstandingCommands() const noexcept {
    return impl_->outstandingCommands->load(std::memory_order_acquire);
}
uint64_t UsdGenExecutionPipeline::CommandCapacity() const noexcept {
    return impl_->commandCapacity;
}
void UsdGenExecutionPipeline::Drain() {
    if (usdGenExecutionGraphActive) throw std::logic_error("graph work must not synchronously drain a pipeline");
    if (UsdGenProcessShutdownInProgress()) return;
    impl_->pending.Wait();
}

void UsdGenExecutionPipeline::Shutdown() {
    Shutdown({});
}

void UsdGenExecutionPipeline::Shutdown(std::function<void()> const& admissionClosed) {
    if (usdGenExecutionGraphActive) throw std::logic_error("graph work must not synchronously shut down a pipeline");
    impl_->closing->store(true, std::memory_order_release);
    if (admissionClosed) {
        try { admissionClosed(); }
        catch (...) { impl_->callbackFailures.fetch_add(1, std::memory_order_relaxed); }
    }
    if (UsdGenProcessShutdownInProgress()) return;
    impl_->pending.Wait();
}

} // namespace usdGen
