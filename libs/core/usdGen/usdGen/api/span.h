// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Non-owning contiguous view (C++17 stand-in for std::span).
#pragma once
#include <cstddef>
#include <type_traits>
#include <utility>

namespace usdGen {

template <class T>
class UsdGenSpan;

namespace span_detail {
template <class C>
struct IsSpan : std::false_type {};
template <class U>
struct IsSpan<UsdGenSpan<U>> : std::true_type {};
/// C's data() as T reads it: const containers and spans of const T use the
/// const overload, so a VtArray never detaches for a read-only view.
template <class T, class C>
using DataOf = decltype(std::declval<std::conditional_t<std::is_const_v<T>, C const &, C &>>().data());
template <class P, class T, class = void>
struct PointsTo : std::false_type {};
template <class P, class T>
struct PointsTo<P, T, std::enable_if_t<std::is_pointer_v<P> && std::is_object_v<std::remove_pointer_t<P>>>>
    : std::is_convertible<std::remove_pointer_t<P> (*)[], T (*)[]> {};
template <class T, class C, class = void>
struct IsContainer : std::false_type {};
template <class T, class C>
struct IsContainer<T, C, std::void_t<DataOf<T, C>, decltype(std::declval<C const &>().size())>>
    : PointsTo<DataOf<T, C>, T> {};
}  // namespace span_detail

template <class T>
class UsdGenSpan {
public:
    constexpr UsdGenSpan() noexcept = default;
    using value_type = std::remove_cv_t<T>;   // lets a host span's container constructor take a UsdGenSpan
    constexpr UsdGenSpan(T *data, size_t size) noexcept : _data(data), _size(size) {}
    template <size_t N>
    constexpr UsdGenSpan(T (&array)[N]) noexcept : _data(array), _size(N) {}
    template <class U, class = std::enable_if_t<std::is_convertible_v<U (*)[], T (*)[]>>>
    constexpr UsdGenSpan(UsdGenSpan<U> o) noexcept : _data(o.data()), _size(o.size()) {}
    /// Any container with data() and size(), as TfSpan: a span of mutable T
    /// needs a mutable lvalue (VtArray detaches); a span of const T reads
    /// through the const data() and also takes a temporary.
    template <class C, class B = std::remove_cv_t<std::remove_reference_t<C>>,
              class = std::enable_if_t<!span_detail::IsSpan<B>::value && !std::is_array_v<B> &&
                                       (std::is_const_v<T> || (std::is_lvalue_reference_v<C> &&
                                                               !std::is_const_v<std::remove_reference_t<C>>)) &&
                                       span_detail::IsContainer<T, std::remove_reference_t<C>>::value>>
    constexpr UsdGenSpan(C &&c) : _data(_Data(c)), _size(c.size()) {}
    constexpr T *data() const noexcept { return _data; }
    constexpr T const *cdata() const noexcept { return _data; }     // host array spelling
    constexpr size_t size() const noexcept { return _size; }
    constexpr bool empty() const noexcept { return _size == 0; }
    constexpr T &operator[](size_t i) const noexcept { return _data[i]; }
    constexpr T *begin() const noexcept { return _data; }
    constexpr T *end() const noexcept { return _data + _size; }
    constexpr T const *cbegin() const noexcept { return _data; }
    constexpr T const *cend() const noexcept { return _data + _size; }
    constexpr T &front() const noexcept { return _data[0]; }
    constexpr T &back() const noexcept { return _data[_size - 1]; }

private:
    template <class C>
    static constexpr T *_Data(C &c)
    {
        if constexpr (std::is_const_v<T>) return std::as_const(c).data();
        else return c.data();
    }

    T *_data = nullptr;
    size_t _size = 0;
};

/// `d.params = UsdGenSpan(kParams);` deduces the element type from a C array.
template <class T, size_t N>
UsdGenSpan(T (&)[N]) -> UsdGenSpan<T>;

}  // namespace usdGen
