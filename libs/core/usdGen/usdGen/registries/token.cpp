// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// The interner: an open-addressed table of atomic rep pointers (linear
// probing, CAS insert) over an append-only bump arena. Lock-free, insert
// only, never freed.
#include "usdGen/api/ext/tokenScope.h"
#include "usdGen/api/status.h"

#include <atomic>
#include <cstdlib>
#include <new>
#include <string>

namespace usdGen {

bool UsdGenOpRegistryFrozenIfBuilt() noexcept;   // opRegistry.cpp

namespace {
std::atomic<uint64_t> gTotalInterns{0};
std::atomic<uint64_t> gExecInterns{0};
std::atomic<bool> gTokenChecks{true};
thread_local int tCompileScopeDepth = 0;

// Table: 2^20 slots; interning stops (empty token + coding error) at 75%.
constexpr size_t kSlotBits = 20;
constexpr size_t kSlots = size_t(1) << kSlotBits;
constexpr size_t kMask = kSlots - 1;
constexpr uint32_t kMaxTokens = uint32_t(kSlots / 4 * 3);
using Slot = std::atomic<UsdGenTokenRep const *>;
std::atomic<Slot *> gTable{nullptr};
std::atomic<uint32_t> gCount{0};
std::atomic<uint32_t> gNextId{0};
std::atomic<bool> gFullReported{false};

// Arena: 64 KiB blocks, the current block's cursor bumped by fetch_add, a
// new block published by CAS. Larger requests get their own allocation.
constexpr size_t kBlockBytes = size_t(64) << 10;
struct Block {
    std::atomic<size_t> used{0};
    alignas(alignof(std::max_align_t)) unsigned char bytes[kBlockBytes];
};
std::atomic<Block *> gBlock{nullptr};

constexpr size_t kAlign = alignof(std::max_align_t);
constexpr size_t AlignUp(size_t n) noexcept { return (n + kAlign - 1) & ~(kAlign - 1); }

void *ArenaAlloc(size_t bytes)
{
    size_t const n = AlignUp(bytes);
    if (n > kBlockBytes / 4) return ::operator new(n);
    for (;;) {
        Block *block = gBlock.load(std::memory_order_acquire);
        if (block) {
            size_t const at = block->used.fetch_add(n, std::memory_order_relaxed);
            if (at + n <= kBlockBytes) return block->bytes + at;
        }
        Block *fresh = new Block;
        fresh->used.store(n, std::memory_order_relaxed);
        if (gBlock.compare_exchange_strong(block, fresh, std::memory_order_acq_rel,
                                           std::memory_order_acquire))
            return fresh->bytes;
        delete fresh;   // another thread published a block; retry in it
    }
}

uint64_t Fnv1a64(std::string_view text) noexcept
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (char c : text) {
        h ^= uint64_t(static_cast<unsigned char>(c));
        h *= 0x100000001b3ull;
    }
    return h;
}

// TfToken's compare code: up to the first NUL, each char widened as plain
// char (so a byte >= 0x80 sign-extends where char is signed, as in Tf).
uint64_t CompareCode(char const *p) noexcept
{
    uint64_t code = 0;
    int n = int(sizeof(code));
    while (n--) {
        code |= static_cast<uint64_t>(*p) << (8 * n);
        if (*p) ++p;
    }
    return code;
}

size_t SlotOf(uint64_t hash) noexcept
{
    uint64_t h = hash ^ (hash >> 31);
    h *= 0x9e3779b97f4a7c15ull;
    return size_t(h >> (64 - kSlotBits));
}

bool Matches(UsdGenTokenRep const *rep, uint64_t hash, std::string_view text) noexcept
{
    return rep->hash == hash && rep->size == text.size() &&
           (text.empty() || std::memcmp(rep->text, text.data(), text.size()) == 0);
}

Slot *Table()
{
    if (Slot *t = gTable.load(std::memory_order_acquire)) return t;
    Slot *fresh = new Slot[kSlots]();
    Slot *winner = nullptr;
    if (gTable.compare_exchange_strong(winner, fresh, std::memory_order_acq_rel,
                                       std::memory_order_acquire))
        return fresh;
    delete[] fresh;
    return winner;
}

UsdGenTokenRep const *NewRep(uint64_t hash, std::string_view text)
{
    // [rep][std::string][text + NUL], all from the arena.
    size_t const repBytes = AlignUp(sizeof(UsdGenTokenRep));
    size_t const strBytes = AlignUp(sizeof(std::string));
    unsigned char *mem = static_cast<unsigned char *>(ArenaAlloc(repBytes + strBytes + text.size() + 1));
    char *chars = reinterpret_cast<char *>(mem + repBytes + strBytes);
    if (!text.empty()) std::memcpy(chars, text.data(), text.size());
    chars[text.size()] = '\0';
    auto *string = new (mem + repBytes) std::string(chars, text.size());
    auto *rep = new (mem) UsdGenTokenRep;
    rep->hash = hash;
    rep->compareCode = CompareCode(chars);
    rep->id = gNextId.fetch_add(1, std::memory_order_relaxed) + 1;
    rep->size = uint32_t(text.size());
    rep->text = chars;
    rep->string = string;
    return rep;
}

std::string const gEmptyString;
}  // namespace

// Called once by the op registry constructor.
void UsdGenReadTokenCheckSetting() noexcept
{
    char const *value = std::getenv("USDGEN_TOKEN_CHECKS");
    gTokenChecks.store(!(value && value[0] == '0' && value[1] == '\0'), std::memory_order_relaxed);
}

std::string const &UsdGenToken::GetString() const noexcept
{
    return _rep ? *_rep->string : gEmptyString;
}

UsdGenToken UsdGenFindToken(std::string_view text) noexcept
{
    if (text.empty()) return UsdGenToken();
    Slot const *table = gTable.load(std::memory_order_acquire);
    if (!table) return UsdGenToken();
    uint64_t const hash = Fnv1a64(text);
    for (size_t i = SlotOf(hash);; i = (i + 1) & kMask) {
        UsdGenTokenRep const *rep = table[i].load(std::memory_order_acquire);
        if (!rep) return UsdGenToken();
        if (Matches(rep, hash, text)) return UsdGenToken(rep);
    }
}

UsdGenToken UsdGenInternToken(std::string_view text)
{
    gTotalInterns.fetch_add(1, std::memory_order_relaxed);
    if (tCompileScopeDepth == 0 && UsdGenOpRegistryFrozenIfBuilt())
        gExecInterns.fetch_add(1, std::memory_order_relaxed);
    if (text.empty()) return UsdGenToken();
    Slot *table = Table();
    uint64_t const hash = Fnv1a64(text);
    UsdGenTokenRep const *mine = nullptr;
    for (size_t i = SlotOf(hash);; i = (i + 1) & kMask) {
        UsdGenTokenRep const *rep = table[i].load(std::memory_order_acquire);
        while (!rep) {
            if (!mine) {
                if (gCount.load(std::memory_order_relaxed) >= kMaxTokens) {
                    if (!gFullReported.exchange(true, std::memory_order_relaxed))
                        UsdGenReport(UsdGenSeverity::CodingError,
                                     "usdGen: the token table is full (%u tokens); raise kSlotBits",
                                     unsigned(kMaxTokens));
                    return UsdGenToken();
                }
                mine = NewRep(hash, text);
            }
            if (table[i].compare_exchange_strong(rep, mine, std::memory_order_acq_rel,
                                                 std::memory_order_acquire)) {
                gCount.fetch_add(1, std::memory_order_relaxed);
                return UsdGenToken(mine);
            }
            // Lost the slot: 'rep' now holds the winner; compare it below.
        }
        if (Matches(rep, hash, text)) return UsdGenToken(rep);   // 'mine' (if any) stays unused
    }
}

uint64_t UsdGenExecInternCount() noexcept
{
    return gExecInterns.load(std::memory_order_relaxed);
}

uint64_t UsdGenTotalInternCount() noexcept
{
    return gTotalInterns.load(std::memory_order_relaxed);
}

bool UsdGenTokenChecksEnabled() noexcept
{
    return gTokenChecks.load(std::memory_order_relaxed);
}

UsdGenCompileTokenScope::UsdGenCompileTokenScope() noexcept { ++tCompileScopeDepth; }
UsdGenCompileTokenScope::~UsdGenCompileTokenScope() { --tCompileScopeDepth; }

}  // namespace usdGen
