// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// UsdGenValue: a closed set of kinds (the rows of usdGenExpr's value-type
// table) plus Boxed payloads that modules register by tag. Type dispatch is
// exact, as VtValue's: IsHolding<T> never converts (1 int != 1.0f). Small
// kinds and arrays live inline; strings, asset paths, path expressions,
// Matrix3d/4d and string arrays live in a refcounted block allocated in
// usdGen. GetHash is process-local: equal values hash equal, +0 == -0.
#pragma once
#include "usdGen/api/cowArray.h"
#include "usdGen/api/export.h"
#include "usdGen/api/token.h"
#include "usdGen/api/epoch.h"
#include "usdGen/api/pool.h"
#include "usdGenMath/math.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace usdGen {

enum class UsdGenValueKind : uint8_t {
    Empty, Bool, UChar, Int, UInt, Int64, UInt64, Half, Float, Double, TimeCode,
    Token, String, AssetPath, PathExpression, Opaque, Group,
    Vec2h, Vec2f, Vec2d, Vec2i, Vec3h, Vec3f, Vec3d, Vec3i, Vec4h, Vec4f, Vec4d, Vec4i,
    Matrix2d, Matrix3d, Matrix4d, Quath, Quatf, Quatd,
    Array,     // ElementKind() names the element
    Boxed      // a module's payload, identified by its tag
};

/// The value table's time code row (GfTimeCode's bytes).
struct UsdGenTimeCode {
    double value = 0.0;
    constexpr UsdGenTimeCode(double v = 0.0) noexcept : value(v) {}
    constexpr double GetValue() const noexcept { return value; }
    friend constexpr bool operator==(UsdGenTimeCode a, UsdGenTimeCode b) noexcept { return a.value == b.value; }
    friend constexpr bool operator!=(UsdGenTimeCode a, UsdGenTimeCode b) noexcept { return !(a == b); }
};

/// An asset path as the host resolved it (SdfAssetPath's three paths).
struct UsdGenAssetPath {
    std::string authored;
    std::string evaluated;
    std::string resolved;
    friend bool operator==(UsdGenAssetPath const &a, UsdGenAssetPath const &b) noexcept
    {
        return a.authored == b.authored && a.evaluated == b.evaluated && a.resolved == b.resolved;
    }
    friend bool operator!=(UsdGenAssetPath const &a, UsdGenAssetPath const &b) noexcept { return !(a == b); }
};

/// A path expression by its text.
struct UsdGenPathExpression {
    std::string text;
    friend bool operator==(UsdGenPathExpression const &a, UsdGenPathExpression const &b) noexcept { return a.text == b.text; }
    friend bool operator!=(UsdGenPathExpression const &a, UsdGenPathExpression const &b) noexcept { return !(a == b); }
};

/// The opaque and group rows: no data, all values equal.
struct UsdGenOpaque {
    friend constexpr bool operator==(UsdGenOpaque, UsdGenOpaque) noexcept { return true; }
    friend constexpr bool operator!=(UsdGenOpaque, UsdGenOpaque) noexcept { return false; }
};
struct UsdGenGroup {
    friend constexpr bool operator==(UsdGenGroup, UsdGenGroup) noexcept { return true; }
    friend constexpr bool operator!=(UsdGenGroup, UsdGenGroup) noexcept { return false; }
};

/// Refcounted payload of a Boxed value. Allocated and freed (virtual
/// destructor) in the module that defines it; that module also creates and
/// reads the values that hold it.
class USDGEN_API UsdGenBoxed : public UsdGenRefCounted {
public:
    virtual uint64_t Hash() const noexcept = 0;
    /// 'o' has the same tag.
    virtual bool Equal(UsdGenBoxed const &o) const noexcept = 0;
};

/// A Boxed payload's tag: FNV-1a 64 of a unique name ("usdGenExpr.outputs").
constexpr uint64_t UsdGenBoxTag(char const *name) noexcept
{
    uint64_t h = 0xcbf29ce484222325ull;
    for (; *name; ++name) {
        h ^= uint64_t(static_cast<unsigned char>(*name));
        h *= 0x100000001b3ull;
    }
    return h;
}

/// Host value interop (tests only; tests/utils/pxrInterop.h specialises it).
/// A specialisation has value = true, static UsdGenValue In(H const &) and
/// static H Out(UsdGenValue const &). The primary admits nothing.
template <class H> struct UsdGenHostValue : std::false_type {};

// Kind, C++ type: held inline, trivially copyable.
#define USDGEN_VALUE_INLINE_KINDS(X)                                                          \
    X(Bool, bool) X(UChar, unsigned char) X(Int, int) X(UInt, unsigned int)                   \
    X(Int64, int64_t) X(UInt64, uint64_t) X(Half, UsdGenHalf) X(Float, float)                 \
    X(Double, double) X(TimeCode, UsdGenTimeCode) X(Token, UsdGenToken)                       \
    X(Opaque, UsdGenOpaque) X(Group, UsdGenGroup)                                             \
    X(Vec2h, UsdGenVec2h) X(Vec2f, UsdGenVec2f) X(Vec2d, UsdGenVec2d) X(Vec2i, UsdGenVec2i)  \
    X(Vec3h, UsdGenVec3h) X(Vec3f, UsdGenVec3f) X(Vec3d, UsdGenVec3d) X(Vec3i, UsdGenVec3i)  \
    X(Vec4h, UsdGenVec4h) X(Vec4f, UsdGenVec4f) X(Vec4d, UsdGenVec4d) X(Vec4i, UsdGenVec4i)  \
    X(Matrix2d, UsdGenMatrix2d) X(Quath, UsdGenQuath) X(Quatf, UsdGenQuatf) X(Quatd, UsdGenQuatd)
// Kind, C++ type: held in a usdGen block.
#define USDGEN_VALUE_HEAP_KINDS(X)                                                            \
    X(String, std::string) X(AssetPath, UsdGenAssetPath) X(PathExpression, UsdGenPathExpression) \
    X(Matrix3d, UsdGenMatrix3d) X(Matrix4d, UsdGenMatrix4d)
// Element kind, C++ element: UsdGenCowArray<element>, held inline.
#define USDGEN_VALUE_ARRAY_ELEMENTS(X)                                                        \
    X(Bool, bool) X(UChar, unsigned char) X(Int, int) X(UInt, unsigned int)                   \
    X(Int64, int64_t) X(UInt64, uint64_t) X(Half, UsdGenHalf) X(Float, float)                 \
    X(Double, double) X(TimeCode, UsdGenTimeCode) X(Token, UsdGenToken)                       \
    X(Vec2h, UsdGenVec2h) X(Vec2f, UsdGenVec2f) X(Vec2d, UsdGenVec2d) X(Vec2i, UsdGenVec2i)  \
    X(Vec3h, UsdGenVec3h) X(Vec3f, UsdGenVec3f) X(Vec3d, UsdGenVec3d) X(Vec3i, UsdGenVec3i)  \
    X(Vec4h, UsdGenVec4h) X(Vec4f, UsdGenVec4f) X(Vec4d, UsdGenVec4d) X(Vec4i, UsdGenVec4i)  \
    X(Matrix2d, UsdGenMatrix2d) X(Matrix3d, UsdGenMatrix3d) X(Matrix4d, UsdGenMatrix4d)      \
    X(Quath, UsdGenQuath) X(Quatf, UsdGenQuatf) X(Quatd, UsdGenQuatd)
// Element kind, C++ element: std::vector<element>, held in a usdGen block.
#define USDGEN_VALUE_VECTOR_ELEMENTS(X) \
    X(String, std::string) X(AssetPath, UsdGenAssetPath) X(PathExpression, UsdGenPathExpression)

namespace value_detail {
enum class Store : uint8_t { Inline, Array, Heap, Box };

template <class T>
struct KindOf {
    static constexpr UsdGenValueKind kind = UsdGenValueKind::Empty;
    static constexpr UsdGenValueKind element = UsdGenValueKind::Empty;
    static constexpr Store store = Store::Inline;
};
#define USDGEN_VALUE_KIND_OF(K, T, E, S)                                                \
    template <> struct KindOf<T> {                                                      \
        static constexpr UsdGenValueKind kind = UsdGenValueKind::K;                     \
        static constexpr UsdGenValueKind element = E;                                   \
        static constexpr Store store = Store::S;                                        \
    };
#define USDGEN_VALUE_INLINE_KIND(K, T) USDGEN_VALUE_KIND_OF(K, T, UsdGenValueKind::Empty, Inline)
#define USDGEN_VALUE_HEAP_KIND(K, T) USDGEN_VALUE_KIND_OF(K, T, UsdGenValueKind::Empty, Heap)
#define USDGEN_VALUE_ARRAY_KIND(K, T) USDGEN_VALUE_KIND_OF(Array, UsdGenCowArray<T>, UsdGenValueKind::K, Array)
#define USDGEN_VALUE_VECTOR_KIND(K, T) USDGEN_VALUE_KIND_OF(Array, std::vector<T>, UsdGenValueKind::K, Heap)
USDGEN_VALUE_INLINE_KINDS(USDGEN_VALUE_INLINE_KIND)
USDGEN_VALUE_HEAP_KINDS(USDGEN_VALUE_HEAP_KIND)
USDGEN_VALUE_ARRAY_ELEMENTS(USDGEN_VALUE_ARRAY_KIND)
USDGEN_VALUE_VECTOR_ELEMENTS(USDGEN_VALUE_VECTOR_KIND)
#undef USDGEN_VALUE_INLINE_KIND
#undef USDGEN_VALUE_HEAP_KIND
#undef USDGEN_VALUE_ARRAY_KIND
#undef USDGEN_VALUE_VECTOR_KIND
#undef USDGEN_VALUE_KIND_OF

template <class T> constexpr bool kHoldable = KindOf<T>::kind != UsdGenValueKind::Empty;

/// A usdGen block holding one heap kind.
template <class T>
struct HeapOf final : UsdGenRefCounted, UsdGenPooled {
    explicit HeapOf(T const &v) : value(v) {}
    T value;
};
}  // namespace value_detail

class UsdGenValue {
public:
    constexpr UsdGenValue() noexcept : _bytes{} {}
    /// Only the closed kinds (and UsdGenCowArray / std::vector of a listed
    /// element); anything else does not convert.
    template <class T, class D = std::decay_t<T>, std::enable_if_t<value_detail::kHoldable<D>, int> = 0>
    UsdGenValue(T &&v) { _Construct<D>(std::forward<T>(v)); }
    /// Text: a String.
    UsdGenValue(char const *s) : UsdGenValue(std::string(s ? s : "")) {}
    /// Host values (tests only, through UsdGenHostValue).
    template <class H, std::enable_if_t<UsdGenHostValue<H>::value, int> = 0>
    UsdGenValue(H const &h) : UsdGenValue(UsdGenHostValue<H>::In(h)) {}
    template <class H, std::enable_if_t<UsdGenHostValue<H>::value, int> = 0>
    operator H() const { return UsdGenHostValue<H>::Out(*this); }

    /// A Boxed value; 'tag' names the payload's type (UsdGenBoxTag).
    static UsdGenValue Box(uint64_t tag, UsdGenRef<const UsdGenBoxed> payload) noexcept
    {
        UsdGenValue v;
        if (!payload) return v;
        UsdGenRefCounted const *p = payload.Detach();
        std::memcpy(v._bytes, &p, sizeof(p));
        std::memcpy(v._bytes + sizeof(p), &tag, sizeof(tag));
        v._kind = UsdGenValueKind::Boxed;
        v._store = value_detail::Store::Box;
        return v;
    }

    UsdGenValue(UsdGenValue const &o) noexcept { _CopyFrom(o); }
    UsdGenValue(UsdGenValue &&o) noexcept { _Steal(o); }
    UsdGenValue &operator=(UsdGenValue const &o) noexcept
    {
        if (this != &o) {
            UsdGenValue tmp(o);
            Swap(tmp);
        }
        return *this;
    }
    UsdGenValue &operator=(UsdGenValue &&o) noexcept
    {
        if (this != &o) {
            UsdGenValue tmp(std::move(o));
            Swap(tmp);
        }
        return *this;
    }
    ~UsdGenValue()
    {
        if (_store != value_detail::Store::Inline) _Release();
    }
    void Swap(UsdGenValue &o) noexcept
    {
        UsdGenValue tmp;
        _Relocate(&tmp, this);
        _Relocate(this, &o);
        _Relocate(&o, &tmp);
        tmp._kind = UsdGenValueKind::Empty;   // its bytes moved on
        tmp._element = UsdGenValueKind::Empty;
        tmp._store = value_detail::Store::Inline;
    }

    bool IsEmpty() const noexcept { return _kind == UsdGenValueKind::Empty; }
    UsdGenValueKind Kind() const noexcept { return _kind; }
    UsdGenValueKind ElementKind() const noexcept { return _element; }
    bool IsArrayValued() const noexcept { return _kind == UsdGenValueKind::Array; }
    USDGEN_API size_t GetArraySize() const noexcept;

    /// Exact type, as VtValue.
    template <class T>
    bool IsHolding() const noexcept
    {
        using K = value_detail::KindOf<T>;
        static_assert(K::kind != UsdGenValueKind::Empty, "not a UsdGenValue kind (Boxed payloads: GetBoxed)");
        return _kind == K::kind && _element == K::element;
    }
    template <class T>
    T const &UncheckedGet() const noexcept
    {
        using K = value_detail::KindOf<T>;
        static_assert(K::kind != UsdGenValueKind::Empty, "not a UsdGenValue kind (Boxed payloads: GetBoxed)");
        if constexpr (K::store == value_detail::Store::Heap)
            return static_cast<value_detail::HeapOf<T> const *>(_Ptr())->value;
        else
            return *std::launder(reinterpret_cast<T const *>(_bytes));
    }
    /// The held T, or T{} when it holds something else.
    template <class T>
    T Get() const
    {
        return IsHolding<T>() ? UncheckedGet<T>() : T{};
    }
    template <class T>
    T GetWithDefault(T const &fallback = T{}) const
    {
        return IsHolding<T>() ? UncheckedGet<T>() : fallback;
    }

    /// Boxed: the payload's tag (0 otherwise).
    uint64_t BoxedTag() const noexcept
    {
        if (_kind != UsdGenValueKind::Boxed) return 0;
        uint64_t tag = 0;
        std::memcpy(&tag, _bytes + sizeof(void *), sizeof(tag));
        return tag;
    }
    /// The payload when it is a T (T derives UsdGenBoxed and has kBoxTag).
    template <class T>
    T const *GetBoxed() const noexcept
    {
        static_assert(std::is_base_of<UsdGenBoxed, T>::value, "T derives UsdGenBoxed");
        if (BoxedTag() != T::kBoxTag) return nullptr;
        return static_cast<T const *>(static_cast<UsdGenBoxed const *>(_Ptr()));
    }

    /// Kind, element and content; +0 == -0. Process-local.
    USDGEN_API uint64_t GetHash() const noexcept;
    /// The Sdf type name ("float3", "token[]"); "" when empty.
    USDGEN_API char const *GetTypeName() const noexcept;
    friend bool operator==(UsdGenValue const &a, UsdGenValue const &b) noexcept { return _Equal(a, b); }
    friend bool operator!=(UsdGenValue const &a, UsdGenValue const &b) noexcept { return !(a == b); }
    /// TfHash and VtValue find it.
    friend size_t hash_value(UsdGenValue const &v) noexcept { return size_t(v.GetHash()); }

private:
    UsdGenRefCounted const *_Ptr() const noexcept
    {
        UsdGenRefCounted const *p = nullptr;
        std::memcpy(&p, _bytes, sizeof(p));
        return p;
    }
    template <class D, class T>
    void _Construct(T &&v)
    {
        using K = value_detail::KindOf<D>;
        if constexpr (K::store == value_detail::Store::Heap) {
            D const &ref = v;
            _SetHeap(K::kind, K::element, &ref);
        } else {
            static_assert(sizeof(D) <= sizeof(_bytes) && alignof(D) <= 8, "inline kind fits");
            ::new (static_cast<void *>(_bytes)) D(std::forward<T>(v));
        }
        _kind = K::kind;
        _element = K::element;
        _store = K::store;
    }
    void _CopyFrom(UsdGenValue const &o) noexcept
    {
        if (o._store == value_detail::Store::Inline) {
            std::memcpy(_bytes, o._bytes, sizeof(_bytes));
            _kind = o._kind;
            _element = o._element;
            _store = o._store;
            return;
        }
        _CopySlow(o);
    }
    void _Steal(UsdGenValue &o) noexcept
    {
        _Relocate(this, &o);
        o._kind = UsdGenValueKind::Empty;
        o._element = UsdGenValueKind::Empty;
        o._store = value_detail::Store::Inline;
    }
    /// Bitwise move (every inline representation is relocatable).
    static void _Relocate(UsdGenValue *to, UsdGenValue *from) noexcept
    {
        std::memcpy(to->_bytes, from->_bytes, sizeof(_bytes));
        to->_kind = from->_kind;
        to->_element = from->_element;
        to->_store = from->_store;
    }
    /// Copies 'src' (the C++ type of kind/element) into a usdGen block.
    USDGEN_API void _SetHeap(UsdGenValueKind kind, UsdGenValueKind element, void const *src);
    USDGEN_API void _CopySlow(UsdGenValue const &o) noexcept;
    USDGEN_API void _Release() noexcept;
    USDGEN_API static bool _Equal(UsdGenValue const &a, UsdGenValue const &b) noexcept;

    alignas(8) unsigned char _bytes[32];   // inline kinds and arrays; else the block (and the box tag)
    UsdGenValueKind _kind = UsdGenValueKind::Empty;
    UsdGenValueKind _element = UsdGenValueKind::Empty;
    value_detail::Store _store = value_detail::Store::Inline;
};

}  // namespace usdGen
