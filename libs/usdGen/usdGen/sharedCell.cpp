// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/sharedCell.h"

#include "usdGen/parkingWord.h"

namespace usdGen {
namespace {

// Free holders. Push and take-all only, so there is no ABA. Constant-
// initialized and trivially destructible: nothing runs at DLL detach.
std::atomic<UsdGenSharedCellHolder *> gFree{nullptr};
// Bumped whenever a UsdGenLazyLeaked slot leaves its sentinel; first callers
// that lost the claim park on it. Constant-initialized, never destroyed.
std::atomic<uint32_t> gLazySequence{0};
// This thread's spare holders (see UsdGenSharedCellAcquire).
thread_local UsdGenSharedCellHolder *tlsSpare = nullptr;

void PushFree(UsdGenSharedCellHolder *first, UsdGenSharedCellHolder *last) noexcept
{
    UsdGenSharedCellHolder *head = gFree.load(std::memory_order_relaxed);
    do { last->freeNext = head; } while (!gFree.compare_exchange_weak(
        head, first, std::memory_order_release, std::memory_order_relaxed));
}

void RecycleDeleter(UsdGenEpochNode *n) noexcept
{
    UsdGenSharedCellRecycle(static_cast<UsdGenSharedCellHolder *>(n));
}

}  // namespace

UsdGenSharedCellHolder *UsdGenSharedCellAcquire(std::shared_ptr<const void> value)
{
    // O(1): pop this thread's spare chain, else take the whole free stack,
    // keep its head and hand the rest back in one CAS. Only when another
    // thread pushed meanwhile does the rest stay on this thread (a thread
    // that exits with spares leaks them, like the free stack itself).
    UsdGenSharedCellHolder *h = tlsSpare;
    if (!h) {
        h = gFree.exchange(nullptr, std::memory_order_acquire);
        if (h && h->freeNext) {
            UsdGenSharedCellHolder *rest = h->freeNext;
            UsdGenSharedCellHolder *expected = nullptr;
            if (!gFree.compare_exchange_strong(expected, rest, std::memory_order_release,
                                               std::memory_order_relaxed))
                tlsSpare = rest;
            h->freeNext = nullptr;
        }
    } else {
        tlsSpare = h->freeNext;
        h->freeNext = nullptr;
    }
    if (!h) h = new UsdGenSharedCellHolder;
    h->raw = value.get();
    h->weak = value;
    h->strong = std::move(value);
    return h;
}

void UsdGenSharedCellRecycle(UsdGenSharedCellHolder *h) noexcept
{
    h->strong.reset();
    h->weak.reset();
    h->raw = nullptr;
    PushFree(h, h);
}

void UsdGenSharedCellRetire(UsdGenSharedCellHolder *h) noexcept
{
    UsdGenEpochRetire(h, &RecycleDeleter);
}

uint32_t UsdGenLazyLeakedSequence() noexcept
{
    return gLazySequence.load(std::memory_order_seq_cst);
}

void UsdGenLazyLeakedPark(uint32_t seen) noexcept
{
    UsdGenParkWhileEqual(gLazySequence, seen);
}

void UsdGenLazyLeakedPublished() noexcept
{
    gLazySequence.fetch_add(1, std::memory_order_seq_cst);
    UsdGenUnparkAll(gLazySequence);
}

}  // namespace usdGen
