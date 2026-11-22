// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/parkingWord.h"

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  pragma comment(lib, "Synchronization.lib")
#elif defined(__linux__)
#  include <linux/futex.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#else
#  include <chrono>
#  include <thread>
#endif

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t),
              "parking needs a plain 32-bit word");

namespace usdGen {

void UsdGenParkWhileEqual(std::atomic<uint32_t> &word, uint32_t expected)
{
#if defined(_WIN32)
    WaitOnAddress(&word, &expected, sizeof(expected), INFINITE);
#elif defined(__linux__)
    syscall(SYS_futex, reinterpret_cast<uint32_t *>(&word),
            FUTEX_WAIT_PRIVATE, expected, nullptr, nullptr, 0);
#else
    // No public address-wait here: poll. Only idle workers reach this.
    if (word.load(std::memory_order_acquire) == expected)
        std::this_thread::sleep_for(std::chrono::microseconds(100));
#endif
}

void UsdGenUnparkAll(std::atomic<uint32_t> &word)
{
#if defined(_WIN32)
    WakeByAddressAll(&word);
#elif defined(__linux__)
    syscall(SYS_futex, reinterpret_cast<uint32_t *>(&word),
            FUTEX_WAKE_PRIVATE, 0x7fffffff, nullptr, nullptr, 0);
#else
    (void)word;
#endif
}

}  // namespace usdGen
