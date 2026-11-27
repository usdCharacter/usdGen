// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// UsdGenSharedCell as the Vulkan backend uses it: shared_ptr publication without atomic shared_ptr. Readers
// on many threads race writers that Store, CompareExchange, Take and Reset;
// every value a reader sees must be alive and intact, and every value must be
// destroyed exactly once.
#include "usdGen/api/sharedCell.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

template <class T>
using VulkanEpochCell = usdGen::UsdGenSharedCell<T>;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "epoch cell check: %s (%d)\n", #x, __LINE__); std::abort(); } } while (false)

namespace {
std::atomic<long> live{0}, made{0};
struct Value {
    explicit Value(uint64_t v) : value(v), check(~v) { live.fetch_add(1); made.fetch_add(1); }
    ~Value() { CHECK(check == ~value); check = 0; live.fetch_sub(1); }
    uint64_t value, check;
};
}

int main() {
    {
        VulkanEpochCell<Value> cell;
        CHECK(!cell.Load() && !cell.Take());
        cell.Store(std::make_shared<Value>(1));
        auto one = cell.Load();
        CHECK(one && one->value == 1);
        CHECK(!cell.CompareExchange(std::make_shared<Value>(9), std::make_shared<Value>(2)));
        CHECK(cell.CompareExchange(one, std::make_shared<Value>(2)));
        CHECK(cell.Load()->value == 2);
        CHECK(cell.Take()->value == 2 && !cell.Load());
        CHECK(cell.CompareExchange({}, one) && cell.Load() == one);
        cell.Reset();
        one.reset();
        usdGen::UsdGenEpochDrain();
    }
    usdGen::UsdGenEpochDrain();
    CHECK(live.load() == 0);

    unsigned const hw = std::thread::hardware_concurrency();
    unsigned const readers = hw > 4 ? hw : 8, writers = 4;
    constexpr int kWrites = 20000;
    {
        VulkanEpochCell<Value> cell(std::make_shared<Value>(0));
        std::atomic<bool> stop{false};
        std::atomic<uint64_t> reads{0};
        std::vector<std::thread> threads;
        for (unsigned r = 0; r != readers; ++r)
            threads.emplace_back([&] {
                uint64_t n = 0;
                while (!stop.load(std::memory_order_acquire)) {
                    if (auto v = cell.Load()) CHECK(v->check == ~v->value);
                    ++n;
                }
                reads.fetch_add(n);
            });
        std::vector<std::thread> writing;
        for (unsigned w = 0; w != writers; ++w)
            writing.emplace_back([&, w] {
                for (int i = 0; i != kWrites; ++i) {
                    uint64_t const id = (uint64_t(w) << 32) | uint64_t(i);
                    switch (i % 4) {
                    case 0: cell.Store(std::make_shared<Value>(id)); break;
                    case 1: { auto seen = cell.Load(); (void)cell.CompareExchange(seen, std::make_shared<Value>(id)); break; }
                    case 2: if (auto v = cell.Take()) CHECK(v->check == ~v->value); break;
                    default: cell.Reset(); break;
                    }
                    if ((i & 255) == 0) usdGen::UsdGenEpochDrain();
                }
            });
        for (auto& t : writing) t.join();
        stop.store(true, std::memory_order_release);
        for (auto& t : threads) t.join();
        CHECK(reads.load() > 0);
        std::printf("epoch cell stress: %u readers, %u writers, %llu reads, %ld values\n",
                    readers, writers, (unsigned long long)reads.load(), made.load());
    }
    for (int i = 0; i != 4; ++i) usdGen::UsdGenEpochDrain();
    CHECK(live.load() == 0);
    std::puts("epoch cell: PASS");
    return 0;
}
