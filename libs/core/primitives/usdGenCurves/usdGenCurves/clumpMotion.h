// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#ifndef USDGEN_CLUMP_MOTION_H
#define USDGEN_CLUMP_MOTION_H

#include "usdGenCurves/types/curveBuffer.h"
#include "usdGenCurves/export.h"
#include "usdGen/api/cowArray.h"
#include "usdGenMath/math.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace usdGen {

struct UsdGenCurveSetView;   // usdGenCurves/types/curveSet.h

struct UsdGenClumpMotionGroup {
    UsdGenVec3f restAnchor;
    uint64_t centerId = 0;
};

// Read-only dense capture data. Copies and capture clones share the immutable
// allocation; a new binding builds a new vector and cannot mutate a prior one.
// The empty vector an unbound array reads; namespace scope, made at load.
template<class T> inline std::vector<T> const kUsdGenClumpEmptyValues{};

template<class T> class UsdGenClumpReadOnlyArray {
public:
    UsdGenClumpReadOnlyArray() = default;
    explicit UsdGenClumpReadOnlyArray(std::vector<T> values)
        : _data(std::make_shared<const std::vector<T>>(std::move(values))),
          _values(_data->data()), _size(_data->size()) {}
    size_t size() const { return _size; }
    size_t capacity() const { return Get().capacity(); }
    bool empty() const { return _size == 0; }
    T const &operator[](size_t i) const { return _values[i]; }
    auto begin() const { return Get().begin(); }
    auto end() const { return Get().end(); }
    T const &front() const { return _values[0]; }
    T const &back() const { return _values[_size - 1]; }
private:
    std::vector<T> const &Get() const {
        return _data ? *_data : kUsdGenClumpEmptyValues<T>;
    }
    std::shared_ptr<const std::vector<T>> _data;
    T const *_values = nullptr;
    size_t _size = 0;
};

struct UsdGenClumpMotionLevel {
    int level = 0;
    // UINT32_MAX means that the curve has no active group at this level.
    UsdGenClumpReadOnlyArray<uint32_t> groupForCurve;
    UsdGenClumpReadOnlyArray<UsdGenClumpMotionGroup> groups;
    UsdGenFloatArray weight; // global CV order; empty for legacy clumpId-only data
    // Owning CoW snapshots keep a reused allocation alive, making cdata()
    // identity an unambiguous cache key even when upstream buffers retire.
    UsdGenIntArray sourceMembership, sourceCenterId;
    UsdGenFloatArray sourceAnchor;
};

struct UsdGenClumpMotion {
    // Numeric level order, independent of lexical plane-name order.
    std::vector<UsdGenClumpMotionLevel> levels;
};

// Resolve the four native planes once at capture time. Legacy clumpId-only
// levels remain visible with no groups or weights; consumers must not infer
// cohesion from an ID alone. A malformed native quartet fails closed.
USDGEN_CURVES_API bool UsdGenBuildClumpMotion(UsdGenCurveBuffer const &buffer,
                            UsdGenClumpMotion *out,
                            std::string *error = nullptr,
                            UsdGenClumpMotion const *previous = nullptr);
/// View overload: copies the planes first, so the previous-capture reuse
/// never hits there.
USDGEN_CURVES_API bool UsdGenBuildClumpMotion(UsdGenCurveSetView const &curves,
                            UsdGenClumpMotion *out,
                            std::string *error = nullptr,
                            UsdGenClumpMotion const *previous = nullptr);

} // namespace usdGen

#endif // USDGEN_CLUMP_MOTION_H
