#ifndef USDGEN_EXECUTION_PIPELINE_H
#define USDGEN_EXECUTION_PIPELINE_H

#include "usdGen/api/export.h"
#include <atomic>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <vector>

namespace usdGen {

// Shared private scheduler. Independent description pipelines use the same
// worker budget; ownership of a description never serializes unrelated ones.
class USDGEN_API UsdGenExecutionRuntime {
public:
    explicit UsdGenExecutionRuntime(int concurrency = 8);
    ~UsdGenExecutionRuntime();
    UsdGenExecutionRuntime(UsdGenExecutionRuntime const&) = delete;
    UsdGenExecutionRuntime& operator=(UsdGenExecutionRuntime const&) = delete;
private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
    friend class UsdGenExecutionPipeline;
    friend class UsdGenExecutionTaskGraph;
};

// Asynchronous command -> work -> publication graph. The short command owner
// keeps accepting requests while a separate work node evaluates. Only work for
// the same pipeline is ordered (e.g. persistent RBF workspace ownership).
// Submission is not a synchronous acceptance acknowledgement. Requests become
// current when accepted by the command owner, in monotonically increasing
// ticket order; out-of-order older arrivals and completions are superseded.
class USDGEN_API UsdGenExecutionPipeline {
public:
    class USDGEN_API CommandTicket {
    public:
        CommandTicket() noexcept;
        ~CommandTicket();
        CommandTicket(CommandTicket&&) noexcept;
        CommandTicket& operator=(CommandTicket&&) noexcept;
        CommandTicket(CommandTicket const&) = delete;
        CommandTicket& operator=(CommandTicket const&) = delete;
        explicit operator bool() const noexcept;
    private:
        struct State;
        std::shared_ptr<State> state_;
        explicit CommandTicket(std::shared_ptr<State> state) noexcept;
        friend class UsdGenExecutionPipeline;
    };
    class USDGEN_API CommandMailbox {
    public:
        CommandMailbox() noexcept;
        ~CommandMailbox();
        CommandMailbox(CommandMailbox const&) noexcept;
        CommandMailbox& operator=(CommandMailbox const&) noexcept;
        CommandMailbox(CommandMailbox&&) noexcept;
        CommandMailbox& operator=(CommandMailbox&&) noexcept;
        explicit operator bool() const noexcept;
    private:
        struct State;
        std::shared_ptr<State> state_;
        explicit CommandMailbox(std::shared_ptr<State> state) noexcept;
        friend class UsdGenExecutionPipeline;
    };
    enum class Outcome { Published, Superseded, Failed };
    struct USDGEN_API Cancellation {
        uint64_t epoch = 0;
        bool Superseded() const noexcept;
    private:
        std::shared_ptr<const std::atomic<uint64_t>> accepted;
        std::shared_ptr<const std::atomic<bool>> closing;
        friend class UsdGenExecutionPipeline;
    };
    using Publish = std::function<void()>;
    // Work captures immutable input values. Return a short publication action
    // owning the completed output; it executes only for the accepted epoch.
    // Work may inspect Cancellation between dependent tasks. Device work must
    // explicitly bind its device and retain buffers until device completion.
    using Work = std::function<Publish(Cancellation const&)>;
    using Completion = std::function<void(uint64_t, Outcome, std::exception_ptr)>;
    // Async work must arrange for its completion itself (for example after a
    // backend event).  The first completion wins; duplicate and late calls
    // are harmless.  A null publication without an error is a failure.
    using AsyncCompletion = std::function<void(Publish, std::exception_ptr)>;
    using AsyncWork = std::function<void(Cancellation const&, AsyncCompletion)>;

    // requestCapacity bounds retained Work payloads from Submit, including
    // owner ingress, running work, and terminal replies. commandCapacity
    // bounds queued owner commands/cancellations and CommandTickets reserved
    // before an external producer starts. The currently executing owner frame
    // releases its credit before invoking user code, so callback reentry is
    // possible even at capacity one.
    explicit UsdGenExecutionPipeline(UsdGenExecutionRuntime& runtime,
                                     uint64_t requestCapacity = 4096,
                                     uint64_t commandCapacity = 4096);
    // Destruction is an external ownership boundary, never a graph callback.
    // Violating this precondition terminates instead of self-deadlocking.
    ~UsdGenExecutionPipeline();
    UsdGenExecutionPipeline(UsdGenExecutionPipeline const&) = delete;
    UsdGenExecutionPipeline& operator=(UsdGenExecutionPipeline const&) = delete;
    // Allocation/dispatch failures may throw; reserved epoch numbers are never
    // reused, so callers must allow gaps. Submission returns 0 when the bounded
    // request ingress is full. A framework dispatch exception from
    // Drain is fatal to this pipeline; retain the last published snapshot and
    // replace the pipeline rather than assuming queued callbacks completed.
    uint64_t Submit(Work work, Completion completion = {});
    uint64_t SubmitAsync(AsyncWork work, Completion completion = {});
    // Non-coalesced short mutation of owner state. Capture input values, not
    // caller-owned mutable references. Commands may enqueue work/commands but
    // must not wait for them. They do not themselves change the work epoch.
    // onCancel runs if shutdown discards an accepted command before execution.
    // Returns false without retaining either callback if command ingress is
    // full. A ticket reserves that bounded ingress before producer handoff;
    // discarded tickets automatically return their admission credit.
    bool PostCommand(std::function<void()> command, std::function<void()> onCancel = {});
    CommandTicket ReserveCommandTicket();
    bool PostCommand(CommandTicket&& ticket, std::function<void()> command,
                     std::function<void()> onCancel = {});
    // Coalescing persistent producer mailbox. It reserves one command credit
    // until all handles and its queued owner message are gone; each post
    // replaces the latest callback and schedules at most one owner message.
    // A framework enqueue failure breaks the mailbox and discards its latest
    // callback; subsequent posts return false.
    CommandMailbox ReserveCommandMailbox();
    bool PostLatestCommand(CommandMailbox const& mailbox, std::function<void()> command);
    bool CommandMailboxBroken(CommandMailbox const& mailbox) const noexcept;
    // Synchronous compatibility boundaries only. Every call owns a separate
    // framework completion graph, so concurrent callers never drain the same
    // graph. The caller parks on its own latch; no mutex/future/spin loop.
    // Both reject calls from ANY pipeline work item or callback.
    void InvokeOwner(std::function<void()> command);
    // dispatch must invoke/retain completion and eventually call it. It may
    // throw only if no asynchronous work retaining caller references was
    // accepted. The completion is idempotent, including immediate completion.
    void Await(std::function<void(std::function<void()>)> dispatch);
    static bool IsExecuting() noexcept;
    // True only in this pipeline's serialized owner frame, not another
    // pipeline's owner or this pipeline's parallel preparation work.
    bool IsExecutingOwner() const noexcept;
    uint64_t CancelPending();
    uint64_t AcceptedEpoch() const noexcept;
    uint64_t CallbackFailures() const noexcept;
    uint64_t OutstandingCommands() const noexcept;
    uint64_t CommandCapacity() const noexcept;
    // External boundary only, one draining caller. Never invoke from a graph
    // callback/work item; callback re-entry uses Submit, which never waits.
    // Before destruction the owner must stop/join external submitters.
    void Drain();
    // Stop acceptance and complete/cancel queued callbacks while caller-owned
    // state still exists. Same external-only/single-waiter rule as Drain.
    void Shutdown();
    // Optional external-boundary observation, after admission closes and
    // before draining. Must be nonblocking and must not access owner state.
    // Observer exceptions are counted, but cannot bypass the shutdown drain.
    void Shutdown(std::function<void()> const& admissionClosed);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Bounded contiguous sequence admission. TryIssue may be called concurrently;
// Complete belongs to one externally scheduled owner. The ring is owner-only:
// frontend code may read counters and issue, but never examines completion
// slots. CAS reserves an available sequence and never waits for another call.
class USDGEN_API UsdGenExecutionSequenceWindow {
public:
    explicit UsdGenExecutionSequenceWindow(uint64_t capacity);
    uint64_t TryIssue() noexcept;
    // Returns false for stale, future, or duplicate completion. A successful
    // out-of-order completion is retained until all earlier sequences finish.
    bool Complete(uint64_t sequence) noexcept;
    uint64_t LastIssued() const noexcept;
    uint64_t CompletedThrough() const noexcept;
    uint64_t Capacity() const noexcept;

private:
    const uint64_t capacity_;
    std::vector<uint64_t> completed_;
    std::atomic<uint64_t> lastIssued_{0};
    std::atomic<uint64_t> completedThrough_{0};
};

} // namespace usdGen
#endif
