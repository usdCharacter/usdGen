// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenVulkan/vulkan/externalRetirement.h"

#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

using namespace usdGen::vulkan;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "Vulkan retirement check: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

namespace {
struct State {
    std::mutex mutex;
    std::condition_variable changed;
    bool entered = false, release = false;
    std::atomic<unsigned> completed{0}, destroyed{0};
};
State* state = nullptr;
struct Capture {
    ~Capture() { state->destroyed.fetch_add(1); }
};
void ReleaseAtExit() {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->release = true;
    state->changed.notify_one();
}
void VerifyAtExit() {
    // Registered before the queue: its exit hook must join both callbacks
    // and destroy their captures before this older exit observer runs.
    CHECK(state->completed.load() == 2 && state->destroyed.load() == 1);
    CHECK(!EnqueueVulkanExternalRetirement([] { CHECK(false); }));
}
void LaterBackendShutdown() {
    // Simulates a driver module initialized after the first retirement. Its
    // teardown must see later accepted native callbacks already joined.
    CHECK(state->completed.load() == 2 && state->destroyed.load() == 1);
    CHECK(!EnqueueVulkanExternalRetirement([] { CHECK(false); }));
}
} // namespace

int main(int argc, char** argv) {
    state = new State;
    if (argc == 2 && std::string(argv[1]) == "--exit") {
        CHECK(std::atexit(VerifyAtExit) == 0);
        auto capture = std::make_shared<Capture>();
        CHECK(EnqueueVulkanExternalRetirement([capture] {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->entered = true;
            state->changed.notify_one();
            state->changed.wait(lock, [] { return state->release; });
            state->completed.fetch_add(1);
        }));
        capture.reset();
        {
            std::unique_lock<std::mutex> lock(state->mutex);
            state->changed.wait(lock, [] { return state->entered; });
        }
        CHECK(EnqueueVulkanExternalRetirement([] { state->completed.fetch_add(1); }));
        CHECK(std::atexit(LaterBackendShutdown) == 0);
        CHECK(RegisterVulkanExternalRetirementAtExit());
        // The accepted callbacks are still unfinished as main returns.
        CHECK(state->completed.load() == 0);
        CHECK(std::atexit(ReleaseAtExit) == 0);
        return 0;
    }
    CHECK(argc == 1);
    auto const caller = std::this_thread::get_id();
    for (unsigned i = 0; i != 256; ++i) {
        CHECK(EnqueueVulkanExternalRetirement([caller, i] {
            CHECK(std::this_thread::get_id() != caller);
            CHECK(state->completed.fetch_add(1) == i);
            if (i == 127) throw 1; // A failed callback cannot strand its successors.
        }));
    }
    CloseVulkanExternalRetirement();
    CHECK(state->completed.load() == 256);
    CHECK(!EnqueueVulkanExternalRetirement([] { CHECK(false); }));
    CloseVulkanExternalRetirement();
    return 0;
}
