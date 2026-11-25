// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Non-owning contiguous view (C++17 stand-in for std::span).
#pragma once
#include <cstddef>
#include <type_traits>

namespace usdGen {

template <class T>
class UsdGenSpan {
public:
    constexpr UsdGenSpan() noexcept = default;
    using value_type = std::remove_cv_t<T>;   // lets TfSpan's container constructor take a UsdGenSpan
    constexpr UsdGenSpan(T *data, size_t size) noexcept : _data(data), _size(size) {}
    template <size_t N>
    constexpr UsdGenSpan(T (&array)[N]) noexcept : _data(array), _size(N) {}
    template <class U, class = std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>>>
    constexpr UsdGenSpan(UsdGenSpan<U> o) noexcept : _data(o.data()), _size(o.size()) {}
    constexpr T *data() const noexcept { return _data; }
    constexpr T const *cdata() const noexcept { return _data; }     // VtArray spelling
    constexpr size_t size() const noexcept { return _size; }
    constexpr bool empty() const noexcept { return _size == 0; }
    constexpr T &operator[](size_t i) const noexcept { return _data[i]; }
    constexpr T *begin() const noexcept { return _data; }
    constexpr T *end() const noexcept { return _data + _size; }
    constexpr T &front() const noexcept { return _data[0]; }
    constexpr T &back() const noexcept { return _data[_size - 1]; }

private:
    T *_data = nullptr;
    size_t _size = 0;
};

/// `d.params = UsdGenSpan(kParams);` deduces the element type from a C array.
template <class T, size_t N>
UsdGenSpan(T (&)[N]) -> UsdGenSpan<T>;

}  // namespace usdGen
