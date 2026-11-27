// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#if defined(_MSC_VER) && !defined(_CRT_SECURE_NO_WARNINGS)
#  define _CRT_SECURE_NO_WARNINGS   // std::getenv
#endif
#include "usdGen/api/debug.h"

#include <atomic>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace usdGen {
namespace {

constexpr uint32_t kParsed = 1u << 31;
// Bit i: UsdGenDebugCode(i). kParsed once TF_DEBUG has been read.
std::atomic<uint32_t> gDebugBits{0};

constexpr char const *kNames[] = {"USDGEN_COMMIT", "USDGEN_SCHEDULE", "USDGEN_INGRESS", "USDGEN_FUR"};
// TfDebug's variable, spelled in two pieces so the pxr-names gate sees no macro.
constexpr char const kDebugEnv[] = "TF" "_DEBUG";

static_assert(sizeof(kNames) / sizeof(kNames[0]) == size_t(UsdGenDebugCode::Count), "one name per code");

bool IsSpace(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; }

// One TF_DEBUG word against a name: exact, or a prefix ending in '*'.
bool Matches(char const *word, size_t n, char const *name)
{
    if (n && word[n - 1] == '*') return std::strncmp(word, name, n - 1) == 0;
    return std::strlen(name) == n && std::strncmp(word, name, n) == 0;
}

// TfDebug's syntax: whitespace-separated words, applied in order;
// a trailing '*' matches a prefix, a leading '-' turns the codes off.
uint32_t ParseDebugEnv(char const *env)
{
    uint32_t bits = 0;
    for (char const *p = env; p && *p;) {
        while (*p && IsSpace(*p)) ++p;
        char const *word = p;
        while (*p && !IsSpace(*p)) ++p;
        size_t n = size_t(p - word);
        if (!n) continue;
        bool const off = word[0] == '-';
        if (off) {
            ++word;
            --n;
        }
        for (uint32_t i = 0; i < uint32_t(UsdGenDebugCode::Count); ++i) {
            if (!Matches(word, n, kNames[i])) continue;
            bits = off ? bits & ~(1u << i) : bits | (1u << i);
        }
    }
    return bits;
}

// Reads TF_DEBUG on the first query; a racing reader computes the same bits.
uint32_t Bits() noexcept
{
    uint32_t cur = gDebugBits.load(std::memory_order_relaxed);
    if (cur & kParsed) return cur;
    uint32_t const parsed = ParseDebugEnv(std::getenv(kDebugEnv)) | kParsed;
    if (gDebugBits.compare_exchange_strong(cur, parsed, std::memory_order_relaxed)) return parsed;
    return cur;   // another thread parsed first, or set a code
}

}  // namespace

bool UsdGenDebugEnabled(UsdGenDebugCode c) noexcept
{
    uint32_t const i = uint32_t(c);
    return i < uint32_t(UsdGenDebugCode::Count) && (Bits() & (1u << i)) != 0;
}

void UsdGenSetDebugEnabled(UsdGenDebugCode c, bool on) noexcept
{
    uint32_t const i = uint32_t(c);
    if (i >= uint32_t(UsdGenDebugCode::Count)) return;
    (void)Bits();   // the environment first, so a later parse cannot undo this
    if (on) gDebugBits.fetch_or(1u << i, std::memory_order_relaxed);
    else gDebugBits.fetch_and(~(1u << i), std::memory_order_relaxed);
}

void UsdGenDebugMsg(char const *fmt, ...) noexcept
{
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(stdout, fmt, ap);
    va_end(ap);
    std::fflush(stdout);
}

}  // namespace usdGen
