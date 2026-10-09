// Morton key contract (usdGenMath/usdGenMath/kernels.h): the magic-mask
// interleave must spread bits exactly like the 21-iteration loop it
// replaced, so scatter's capture order is unchanged. Proven by
// exhaustive comparison over the full 2^21 input domain per axis, plus
// wide-input masking, plus UsdGenMortonKey3 edge properties (origin
// golden, clamp saturation, signed-zero indifference).
#include "usdGenMath/usdGenMath/kernels.h"

#include <cstdint>
#include <cstdio>
#include <string>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what, int line)
{
    if (!ok) {
        ++g_failures;
        std::printf("FAIL (line %d): %s\n", line, what.c_str());
    }
}
#define CHECK(cond, what) Check(static_cast<bool>(cond), what, __LINE__)

// The pre-mask loop form, verbatim: the exactness oracle.
uint64_t LoopSplit3(uint32_t v)
{
    v &= 0x001FFFFFu;
    uint64_t s = 0;
    for (int i = 0; i < 21; ++i)
        s |= uint64_t((v >> i) & 1u) << (3 * i);
    return s;
}
uint64_t LoopInterleave(uint32_t x, uint32_t y, uint32_t z)
{
    return LoopSplit3(x) | (LoopSplit3(y) << 1) | (LoopSplit3(z) << 2);
}

void CheckExhaustive()
{
    // Every 21-bit value on each axis (the other two at 0, then at max:
    // axes interleave independently, so single-axis sweeps cover split3
    // fully and the shifts/OR are exercised by the max pairing).
    for (uint32_t v = 0; v < (1u << 21); ++v) {
        if (UsdGenMortonInterleave(v, 0, 0) != LoopInterleave(v, 0, 0) ||
            UsdGenMortonInterleave(0, v, 0) != LoopInterleave(0, v, 0) ||
            UsdGenMortonInterleave(0, 0, v) != LoopInterleave(0, 0, v) ||
            UsdGenMortonInterleave(v, 0x1FFFFFu, 0x1FFFFFu) !=
                LoopInterleave(v, 0x1FFFFFu, 0x1FFFFFu)) {
            CHECK(false, "magic-mask interleave matches the loop form");
            return;
        }
    }
    CHECK(true, "magic-mask interleave matches the loop form");
    // Wide inputs mask to 21 bits, as before.
    uint64_t z = 0x123456789abcdefull;
    for (int k = 0; k < 1000000; ++k) {
        z += 0x9E3779B97F4A7C15ull;
        uint32_t x = uint32_t(z >> 17);
        uint32_t y = uint32_t(z >> 41);
        uint32_t w = uint32_t(z);
        if (UsdGenMortonInterleave(x, y, w) != LoopInterleave(x, y, w)) {
            CHECK(false, "wide inputs mask to 21 bits");
            return;
        }
    }
    CHECK(true, "wide inputs mask to 21 bits");
}

void CheckKey3()
{
    // Origin: quant(0) = 2^20 per axis, so bit 20 spreads to bit 60/61/62.
    CHECK(UsdGenMortonKey3(0.0f, 0.0f, 0.0f, 64.0f) == 0x7000000000000000ull,
          "origin key golden");
    CHECK(UsdGenMortonKey3(-0.0f, -0.0f, -0.0f, 64.0f) ==
              UsdGenMortonKey3(0.0f, 0.0f, 0.0f, 64.0f),
          "signed zero keys like zero");
    // Clamp saturation: out-of-window coords stick at the window edges.
    uint64_t const hi = UsdGenMortonKey3(1e30f, 1e30f, 1e30f, 64.0f);
    uint64_t const lo = UsdGenMortonKey3(-1e30f, -1e30f, -1e30f, 64.0f);
    CHECK(hi == UsdGenMortonKey3(1e20f, 1e20f, 1e20f, 64.0f),
          "large coords saturate high");
    CHECK(lo == UsdGenMortonKey3(-1e20f, -1e20f, -1e20f, 64.0f),
          "large coords saturate low");
    CHECK(hi == UsdGenMortonInterleave(0x1FFFFFu, 0x1FFFFFu, 0x1FFFFFu),
          "high clamp is the window max");
    CHECK(lo == 0u, "low clamp is zero");
    // Determinism across repeat calls.
    CHECK(UsdGenMortonKey3(1.5f, -2.25f, 0.125f, 64.0f) ==
              UsdGenMortonKey3(1.5f, -2.25f, 0.125f, 64.0f),
          "key3 is deterministic");
}

}  // namespace

int main()
{
    CheckExhaustive();
    CheckKey3();
    std::printf("testUsdGenMorton: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
