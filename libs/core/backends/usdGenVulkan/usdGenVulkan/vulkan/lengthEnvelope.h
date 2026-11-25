// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_LENGTH_ENVELOPE_H
#define USDGEN_VULKAN_LENGTH_ENVELOPE_H

namespace usdGen::vulkan {
inline float ResolveLengthEnvelope(float blend, float amount) noexcept {
    return float(double(blend) * double(amount));
}
} // namespace usdGen::vulkan

#endif
