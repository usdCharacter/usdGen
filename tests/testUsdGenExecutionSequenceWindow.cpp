#include "usdGen/exec/executionSequenceWindow.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <thread>
#include <vector>

using usdGen::UsdGenExecutionSequenceWindow;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c); return 1; } } while(false)

int main() {
    bool invalid = false;
    try { UsdGenExecutionSequenceWindow bad(0); }
    catch (std::invalid_argument const&) { invalid = true; }
    CHECK(invalid);

    UsdGenExecutionSequenceWindow one(1);
    CHECK(one.Capacity() == 1 && one.TryIssue() == 1 && one.TryIssue() == 0);
    CHECK(!one.Complete(0) && !one.Complete(2));
    CHECK(one.Complete(1) && one.CompletedThrough() == 1);
    CHECK(!one.Complete(1) && one.TryIssue() == 2 && one.Complete(2));

    UsdGenExecutionSequenceWindow gap(3);
    CHECK(gap.TryIssue() == 1 && gap.TryIssue() == 2 && gap.TryIssue() == 3);
    CHECK(gap.Complete(3) && gap.Complete(2));
    CHECK(gap.CompletedThrough() == 0 && gap.TryIssue() == 0);
    CHECK(gap.Complete(1) && gap.CompletedThrough() == 3);
    CHECK(gap.TryIssue() == 4 && gap.TryIssue() == 5 && gap.TryIssue() == 6);
    CHECK(gap.Complete(5) && gap.Complete(4) && gap.CompletedThrough() == 5);
    CHECK(gap.Complete(6) && gap.CompletedThrough() == 6);

    UsdGenExecutionSequenceWindow concurrent(64);
    std::vector<uint64_t> issued(64);
    std::atomic<unsigned> next{0};
    std::vector<std::thread> workers;
    for (unsigned t=0; t<8; ++t) workers.emplace_back([&] {
        for (;;) {
            const auto sequence = concurrent.TryIssue();
            if (!sequence) return;
            const auto index = next.fetch_add(1, std::memory_order_relaxed);
            issued[index] = sequence;
        }
    });
    for (auto& worker : workers) worker.join();
    CHECK(next == 64 && concurrent.TryIssue() == 0 && concurrent.LastIssued() == 64);
    std::vector<bool> seen(65, false);
    for (auto sequence : issued) { CHECK(sequence >= 1 && sequence <= 64 && !seen[sequence]); seen[sequence] = true; }
    for (uint64_t sequence=64; sequence; --sequence) CHECK(concurrent.Complete(sequence));
    CHECK(concurrent.CompletedThrough() == 64);

    UsdGenExecutionSequenceWindow wrap(3);
    for (uint64_t base=1; base<=3000; base += 3) {
        CHECK(wrap.TryIssue() == base && wrap.TryIssue() == base + 1 && wrap.TryIssue() == base + 2);
        CHECK(wrap.TryIssue() == 0);
        CHECK(wrap.Complete(base + 2) && wrap.Complete(base + 1) && wrap.Complete(base));
        CHECK(wrap.CompletedThrough() == base + 2);
        CHECK(!wrap.Complete(base));
    }

    // Concurrent issuers race a single scheduled completion owner. The owner
    // intentionally completes newest-ready entries first, retaining gaps in
    // the fixed ring while producers reuse capacity behind each released
    // prefix. No producer may observe a permanent false-full condition.
    constexpr uint64_t stressTotal = 2000;
    UsdGenExecutionSequenceWindow stress(7);
    std::vector<std::atomic<uint64_t>> ready(stressTotal + 1);
    for (auto& value : ready) value.store(0, std::memory_order_relaxed);
    std::atomic<uint64_t> claims{0};
    std::atomic<bool> stressFailure{false}, stopStress{false};
    workers.clear();
    for (unsigned t=0; t<6; ++t) workers.emplace_back([&] {
        for (;;) {
            const auto claim = claims.fetch_add(1, std::memory_order_relaxed);
            if (claim >= stressTotal) return;
            uint64_t sequence = 0;
            while ((sequence = stress.TryIssue()) == 0) {
                if (stopStress.load(std::memory_order_acquire)) return;
                std::this_thread::yield();
            }
            ready[sequence].store(sequence, std::memory_order_release);
        }
    });
    std::thread owner([&] {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (stress.CompletedThrough() != stressTotal) {
            const auto last = stress.LastIssued();
            const auto through = stress.CompletedThrough();
            for (uint64_t sequence=last; sequence > through; --sequence) {
                if (ready[sequence].exchange(0, std::memory_order_acq_rel) == sequence)
                    if (!stress.Complete(sequence)) { stressFailure = true; stopStress = true; return; }
            }
            if (std::chrono::steady_clock::now() >= deadline) {
                stressFailure = true; stopStress = true; return;
            }
            std::this_thread::yield();
        }
    });
    for (auto& worker : workers) worker.join();
    owner.join();
    CHECK(!stressFailure && stress.LastIssued() == stressTotal &&
          stress.CompletedThrough() == stressTotal);
    return 0;
}
