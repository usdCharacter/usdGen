#ifndef USDGEN_TEST_FLOAT_ULP_FIXTURE_H
#define USDGEN_TEST_FLOAT_ULP_FIXTURE_H

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

// Test-only comparison for computed float geometry, not immutable COW bytes.
// Counts representable float steps, so tolerating interpolation roundoff does
// not introduce a broad absolute epsilon around zero or accept nonfinite data.
inline bool FloatBytesWithinUlps(void const* actual, void const* expected,
                                size_t bytes, uint32_t maxUlps) {
    if (bytes % sizeof(float) || (bytes && (!actual || !expected))) return false;
    auto const* a = static_cast<unsigned char const*>(actual);
    auto const* b = static_cast<unsigned char const*>(expected);
    for (size_t i = 0; i < bytes; i += sizeof(float)) {
        float x, y;
        std::memcpy(&x, a + i, sizeof(x));
        std::memcpy(&y, b + i, sizeof(y));
        if (!std::isfinite(x) || !std::isfinite(y)) return false;
        if (x == y) continue;
        uint32_t u, v;
        std::memcpy(&u, &x, sizeof(u));
        std::memcpy(&v, &y, sizeof(v));
        auto ordered = [](uint32_t bits) {
            return bits & 0x80000000u ? ~bits : bits | 0x80000000u;
        };
        u = ordered(u); v = ordered(v);
        uint32_t distance = u > v ? u - v : v - u;
        if (distance > maxUlps) return false;
    }
    return true;
}

#endif
