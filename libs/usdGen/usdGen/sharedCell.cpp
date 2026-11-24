// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/sharedCell.h"

namespace usdGen {
namespace {

// Free holders. Push and take-all only, so there is no ABA. Constant-
// initialized and trivially destructible: nothing runs at DLL detach.
std::atomic<UsdGenSharedCellHolder *> gFree{nullptr};

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
    UsdGenSharedCellHolder *h = gFree.exchange(nullptr, std::memory_order_acquire);
    if (h) {
        if (UsdGenSharedCellHolder *rest = h->freeNext) {
            UsdGenSharedCellHolder *last = rest;
            while (last->freeNext) last = last->freeNext;
            PushFree(rest, last);
        }
        h->freeNext = nullptr;
    } else {
        h = new UsdGenSharedCellHolder;
    }
    h->weak = value;
    h->strong = std::move(value);
    return h;
}

void UsdGenSharedCellRecycle(UsdGenSharedCellHolder *h) noexcept
{
    h->strong.reset();
    h->weak.reset();
    PushFree(h, h);
}

void UsdGenSharedCellRetire(UsdGenSharedCellHolder *h) noexcept
{
    UsdGenEpochRetire(h, &RecycleDeleter);
}

}  // namespace usdGen
