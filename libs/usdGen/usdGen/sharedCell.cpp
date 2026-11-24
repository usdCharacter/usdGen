// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/sharedCell.h"

namespace usdGen {
namespace {

// Free holders. Push, take-all and install-if-empty only, so there is no
// ABA. Constant-initialized and trivially destructible: nothing runs at DLL
// detach.
UsdGenTreiberStack<UsdGenSharedCellHolder, &UsdGenSharedCellHolder::freeNext> gFree;
// This thread's spare holders (see UsdGenSharedCellAcquire).
thread_local UsdGenSharedCellHolder *tlsSpare = nullptr;

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
        h = gFree.PopAll();
        if (h && h->freeNext) {
            UsdGenSharedCellHolder *rest = h->freeNext;
            if (!gFree.InstallIfEmpty(rest)) tlsSpare = rest;
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
    gFree.Push(h);
}

void UsdGenSharedCellRetire(UsdGenSharedCellHolder *h) noexcept
{
    UsdGenEpochRetire(h, &RecycleDeleter);
}

}  // namespace usdGen
