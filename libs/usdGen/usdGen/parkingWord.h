// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — parking on a 32-bit word, the engine's only blocking primitive.
// A futex on Linux, WaitOnAddress on Windows. Waiting compares the word in
// the kernel, so a wake that lands between the caller's check and its park
// is never lost, and no mutex or condition variable is involved.
#pragma once

#include <atomic>
#include <cstdint>

namespace usdGen {

/// Block while `word` still holds `expected`. May return spuriously; callers
/// re-check in a loop.
void UsdGenParkWhileEqual(std::atomic<uint32_t> &word, uint32_t expected);

/// The same, giving up after about `timeoutMs`. Returns false on timeout.
bool UsdGenParkWhileEqualFor(std::atomic<uint32_t> &word, uint32_t expected,
                             uint32_t timeoutMs);

/// Wake every thread parked on `word`. Call after changing it.
void UsdGenUnparkAll(std::atomic<uint32_t> &word);

}  // namespace usdGen
