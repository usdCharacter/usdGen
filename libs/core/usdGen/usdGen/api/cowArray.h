// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenCowArray<T>: a copy-on-write array of trivially copyable T with
// VtArray's read API and detach rule: a mutator copies exactly when the
// array is not unique (shared, external or a published column). Storage is
// a one-column UsdGenArena from the pools (allocated and freed in usdGen);
// an external arena adopts host memory without copying.
#pragma once
#include "usdGen/api/arena.h"
#include "usdGen/api/token.h"
#include "usdGenMath/math.h"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

namespace usdGen {

template <class T>
class UsdGenCowArray;

/// Host array interop (tests only; tests/utils/pxrInterop.h specialises it).
/// A specialisation has value = true, Element (the engine element type),
/// static UsdGenCowArray<Element> In(H const &) and static H Out(UsdGenCowArray<Element> const &).
/// The primary admits nothing.
template <class H> struct UsdGenHostArray : std::false_type { using Element = void; };

namespace cow_detail {
template <class T> inline size_t ElemHash(T const &v) noexcept
{
    if constexpr (std::is_floating_point_v<T>) {
        T z = v == T(0) ? T(0) : v;   // +0 == -0
        uint64_t b = 0;
        std::memcpy(&b, &z, sizeof(T));
        return size_t(b);
    } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
        return size_t(uint64_t(v));
    } else {
        return hash_value(v);
    }
}
template <class H, class T>
constexpr bool kHostArrayOf = UsdGenHostArray<H>::value && std::is_same_v<typename UsdGenHostArray<H>::Element, T>;
}  // namespace cow_detail

template <class T>
class UsdGenCowArray {
    static_assert(std::is_trivially_copyable_v<T>, "UsdGenCowArray holds trivially copyable T");

public:
    using value_type = T;
    using ElementType = T;
    using size_type = size_t;
    using difference_type = std::ptrdiff_t;
    using reference = T &;
    using const_reference = T const &;
    using pointer = T *;
    using const_pointer = T const *;
    using iterator = T *;
    using const_iterator = T const *;
    using reverse_iterator = std::reverse_iterator<iterator>;
    using const_reverse_iterator = std::reverse_iterator<const_iterator>;

    UsdGenCowArray() noexcept = default;
    /// Value-initialised, as VtArray(n).
    explicit UsdGenCowArray(size_t n)
    {
        if (!n) return;
        _Allocate(n, n);
        for (size_t i = 0; i < n; ++i) ::new (static_cast<void *>(_data + i)) T();
    }
    UsdGenCowArray(size_t n, T const &v)
    {
        if (!n) return;
        _Allocate(n, n);
        std::uninitialized_fill_n(_data, n, v);
    }
    UsdGenCowArray(std::initializer_list<T> il) { _AssignCopy(il.begin(), il.size()); }
    template <class It, class = typename std::iterator_traits<It>::iterator_category>
    UsdGenCowArray(It first, It last) { assign(first, last); }
    /// A host array (tests only, through UsdGenHostArray).
    template <class H, std::enable_if_t<cow_detail::kHostArrayOf<H, T>, int> = 0>
    UsdGenCowArray(H const &h) : UsdGenCowArray(UsdGenHostArray<H>::In(h)) {}
    template <class H, std::enable_if_t<cow_detail::kHostArrayOf<H, T>, int> = 0>
    operator H() const { return UsdGenHostArray<H>::Out(*this); }

    /// A view of 'n' elements of an arena column; copies share the arena.
    static UsdGenCowArray Share(UsdGenRef<const UsdGenArena> arena, T const *data, size_t n) noexcept
    {
        UsdGenCowArray a;
        if (!n) return a;
        a._arena = std::move(arena);
        a._data = const_cast<T *>(data);
        a._size = n;
        return a;
    }
    /// Zero copy over host memory; 'owner' is released with the last copy.
    static UsdGenCowArray Adopt(T const *data, size_t n, UsdGenExternal *owner)
    {
        UsdGenCowArray a;
        if (!n) {
            if (owner) owner->Release();
            return a;
        }
        a._arena = UsdGenArena::Adopt(data, n * sizeof(T), owner);
        a._data = const_cast<T *>(data);
        a._size = n;
        return a;
    }

    UsdGenCowArray(UsdGenCowArray const &o) noexcept : _arena(o._arena), _data(o._data), _size(o._size) {}
    UsdGenCowArray(UsdGenCowArray &&o) noexcept
        : _arena(std::move(o._arena)), _data(std::exchange(o._data, nullptr)), _size(std::exchange(o._size, 0)) {}
    UsdGenCowArray &operator=(UsdGenCowArray const &o) noexcept
    {
        UsdGenCowArray(o).swap(*this);
        return *this;
    }
    UsdGenCowArray &operator=(UsdGenCowArray &&o) noexcept
    {
        UsdGenCowArray(std::move(o)).swap(*this);
        return *this;
    }
    UsdGenCowArray &operator=(std::initializer_list<T> il)
    {
        assign(il);
        return *this;
    }
    ~UsdGenCowArray() = default;

    size_t size() const noexcept { return _size; }
    bool empty() const noexcept { return _size == 0; }
    size_t max_size() const noexcept { return size_t(-1) / sizeof(T); }
    size_t capacity() const noexcept
    {
        if (!_arena) return 0;
        if (_arena->IsExternal()) return _size;
        return size_t(_arena->Payload() + _arena->Capacity() - reinterpret_cast<std::byte const *>(_data)) / sizeof(T);
    }
    T const *cdata() const noexcept { return _data; }
    T const *data() const noexcept { return _data; }
    T *data()
    {
        _Detach();
        return _data;
    }
    T const &operator[](size_t i) const noexcept { return _data[i]; }
    T &operator[](size_t i)
    {
        _Detach();
        return _data[i];
    }
    const_iterator begin() const noexcept { return _data; }
    const_iterator end() const noexcept { return _data + _size; }
    iterator begin()
    {
        _Detach();
        return _data;
    }
    iterator end()
    {
        _Detach();
        return _data + _size;
    }
    const_iterator cbegin() const noexcept { return _data; }
    const_iterator cend() const noexcept { return _data + _size; }
    const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
    const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
    reverse_iterator rbegin() { return reverse_iterator(end()); }
    reverse_iterator rend() { return reverse_iterator(begin()); }
    const_reverse_iterator crbegin() const noexcept { return rbegin(); }
    const_reverse_iterator crend() const noexcept { return rend(); }
    T const &front() const noexcept { return _data[0]; }
    T const &back() const noexcept { return _data[_size - 1]; }
    T &front() { return data()[0]; }
    T &back() { return data()[_size - 1]; }

    /// New elements are value-initialised.
    void resize(size_t n)
    {
        size_t const old = _Resize(n);
        for (size_t i = old; i < n; ++i) ::new (static_cast<void *>(_data + i)) T();
    }
    void resize(size_t n, T const &v)
    {
        size_t const old = _Resize(n);
        if (n > old) std::uninitialized_fill(_data + old, _data + n, v);
    }
    /// VtArray's fill form: fill(first, last) writes the new elements.
    template <class Fill, class = decltype(std::declval<Fill &>()(std::declval<T *>(), std::declval<T *>()))>
    void resize(size_t n, Fill &&fill)
    {
        size_t const old = _Resize(n);
        if (n > old) std::forward<Fill>(fill)(_data + old, _data + n);
    }
    void reserve(size_t n)
    {
        if (n <= capacity()) return;
        _Reallocate(n);
    }
    void clear()
    {
        if (_IsWritable()) _size = 0;
        else UsdGenCowArray().swap(*this);
    }
    void push_back(T const &v)
    {
        T const copy = v;   // v may live in this array
        _Grow(_size + 1);
        _data[_size++] = copy;
    }
    template <class... A>
    T &emplace_back(A &&...a)
    {
        T value(std::forward<A>(a)...);
        _Grow(_size + 1);
        _data[_size] = value;
        return _data[_size++];
    }
    void pop_back()
    {
        _Detach();
        --_size;
    }
    iterator erase(const_iterator pos) { return erase(pos, pos + 1); }
    iterator erase(const_iterator first, const_iterator last)
    {
        size_t const i = size_t(first - _data), j = size_t(last - _data);
        _Detach();
        if (j > i) std::memmove(static_cast<void *>(_data + i), _data + j, (_size - j) * sizeof(T));
        _size -= j - i;
        return _data + i;
    }
    iterator insert(const_iterator pos, T const &v)
    {
        size_t const i = size_t(pos - _data);
        T const copy = v;
        _Grow(_size + 1);
        std::memmove(static_cast<void *>(_data + i + 1), _data + i, (_size - i) * sizeof(T));
        _data[i] = copy;
        ++_size;
        return _data + i;
    }
    /// The range may alias this array.
    template <class It, class = typename std::iterator_traits<It>::iterator_category>
    iterator insert(const_iterator pos, It first, It last)
    {
        size_t const i = size_t(pos - _data);
        UsdGenCowArray const items(first, last);
        size_t const n = items.size();
        if (!n) {
            _Detach();
            return _data + i;
        }
        _Grow(_size + n);
        std::memmove(static_cast<void *>(_data + i + n), _data + i, (_size - i) * sizeof(T));
        std::memcpy(static_cast<void *>(_data + i), items.cdata(), n * sizeof(T));
        _size += n;
        return _data + i;
    }
    iterator insert(const_iterator pos, std::initializer_list<T> il) { return insert(pos, il.begin(), il.end()); }
    iterator insert(const_iterator pos, size_t n, T const &v)
    {
        UsdGenCowArray const items(n, v);
        return insert(pos, items.cbegin(), items.cend());
    }
    void assign(size_t n, T const &v)
    {
        UsdGenCowArray a(n, v);
        swap(a);
    }
    template <class It, class = typename std::iterator_traits<It>::iterator_category>
    void assign(It first, It last)
    {
        if constexpr (std::is_base_of_v<std::forward_iterator_tag,
                                        typename std::iterator_traits<It>::iterator_category>) {
            size_t const n = size_t(std::distance(first, last));
            UsdGenCowArray a;
            if (n) {
                a._Allocate(n, n);
                std::uninitialized_copy(first, last, a._data);
            }
            swap(a);
        } else {
            UsdGenCowArray a;
            for (; first != last; ++first) a.push_back(*first);
            swap(a);
        }
    }
    void assign(std::initializer_list<T> il)
    {
        UsdGenCowArray a(il);
        swap(a);
    }
    void swap(UsdGenCowArray &o) noexcept
    {
        _arena.Swap(o._arena);
        std::swap(_data, o._data);
        std::swap(_size, o._size);
    }

    /// Sole owner of a pool arena: a mutator writes in place.
    bool IsUnique() const noexcept { return _IsWritable(); }
    /// Same arena, data and size.
    bool IsIdentical(UsdGenCowArray const &o) const noexcept
    {
        return _arena == o._arena && _data == o._data && _size == o._size;
    }
    /// Identity (device mirror, caches); null when empty.
    UsdGenArena const *Arena() const noexcept { return _arena.Get(); }
    UsdGenRef<const UsdGenArena> const &ArenaRef() const noexcept { return _arena; }

    friend bool operator==(UsdGenCowArray const &a, UsdGenCowArray const &b) noexcept
    {
        if (a.IsIdentical(b)) return true;
        if (a._size != b._size) return false;
        for (size_t i = 0; i < a._size; ++i)
            if (!(a._data[i] == b._data[i])) return false;
        return true;
    }
    friend bool operator!=(UsdGenCowArray const &a, UsdGenCowArray const &b) noexcept { return !(a == b); }
    friend void swap(UsdGenCowArray &a, UsdGenCowArray &b) noexcept { a.swap(b); }
    /// Process-local (TfHash finds it): FNV-1a over the size and element hashes.
    friend size_t hash_value(UsdGenCowArray const &a) noexcept
    {
        uint64_t h = 0xcbf29ce484222325ull;
        auto mix = [&h](uint64_t v) {
            for (int i = 0; i < 8; ++i) { h ^= (v >> (8 * i)) & 0xffu; h *= 0x100000001b3ull; }
        };
        mix(a._size);
        for (size_t i = 0; i < a._size; ++i) mix(uint64_t(cow_detail::ElemHash(a._data[i])));
        return size_t(h);
    }

private:
    bool _IsWritable() const noexcept
    {
        if (!_arena || _arena->IsExternal() || _arena->IsShared() || _arena->RefCount() != 1) return false;
        std::atomic_thread_fence(std::memory_order_acquire);   // pairs with the other owners' release
        return true;
    }
    void _Allocate(size_t n, size_t cap)
    {
        UsdGenRef<UsdGenArena> arena = UsdGenArena::Acquire(cap * sizeof(T), alignof(T) > 16 ? alignof(T) : 16);
        _data = reinterpret_cast<T *>(arena->Payload());
        _arena = std::move(arena);
        _size = n;
    }
    void _AssignCopy(T const *src, size_t n)
    {
        if (!n) return;
        _Allocate(n, n);
        std::memcpy(static_cast<void *>(_data), src, n * sizeof(T));
    }
    void _Reallocate(size_t cap)
    {
        UsdGenCowArray a;
        a._Allocate(_size, cap);
        if (_size) std::memcpy(static_cast<void *>(a._data), _data, _size * sizeof(T));
        swap(a);
    }
    /// VtArray's detach: copy at the current size when not unique.
    void _Detach()
    {
        if (_data && !_IsWritable()) _Reallocate(_size);
    }
    void _Grow(size_t n)
    {
        if (_IsWritable() && n <= capacity()) return;
        size_t cap = _IsWritable() || !_size ? (n > 2 * _size ? n : 2 * _size) : n;
        if (cap < n) cap = n;
        _Reallocate(cap);
    }
    /// Sizes to n; returns the old size (elements past it are raw).
    size_t _Resize(size_t n)
    {
        size_t const old = _size;
        if (n == old) return old;
        if (!n) {
            clear();
            return old;
        }
        if (_IsWritable() && n <= capacity()) {
            _size = n;
            return old;
        }
        UsdGenCowArray a;
        a._Allocate(n, n);
        size_t const keep = old < n ? old : n;
        if (keep) std::memcpy(static_cast<void *>(a._data), _data, keep * sizeof(T));
        swap(a);
        return old;
    }

    UsdGenRef<const UsdGenArena> _arena;
    T *_data = nullptr;
    size_t _size = 0;
};

using UsdGenFloatArray = UsdGenCowArray<float>;
using UsdGenIntArray = UsdGenCowArray<int>;
using UsdGenBoolArray = UsdGenCowArray<bool>;
using UsdGenUInt64Array = UsdGenCowArray<uint64_t>;
using UsdGenTokenArray = UsdGenCowArray<UsdGenToken>;
using UsdGenVec2fArray = UsdGenCowArray<UsdGenVec2f>;
using UsdGenVec3fArray = UsdGenCowArray<UsdGenVec3f>;
using UsdGenVec4fArray = UsdGenCowArray<UsdGenVec4f>;
using UsdGenVec2dArray = UsdGenCowArray<UsdGenVec2d>;
using UsdGenVec3dArray = UsdGenCowArray<UsdGenVec3d>;
using UsdGenVec4dArray = UsdGenCowArray<UsdGenVec4d>;
using UsdGenVec2hArray = UsdGenCowArray<UsdGenVec2h>;
using UsdGenVec3hArray = UsdGenCowArray<UsdGenVec3h>;
using UsdGenVec4hArray = UsdGenCowArray<UsdGenVec4h>;
using UsdGenVec2iArray = UsdGenCowArray<UsdGenVec2i>;
using UsdGenVec3iArray = UsdGenCowArray<UsdGenVec3i>;
using UsdGenVec4iArray = UsdGenCowArray<UsdGenVec4i>;
using UsdGenMatrix4dArray = UsdGenCowArray<UsdGenMatrix4d>;
}  // namespace usdGen
