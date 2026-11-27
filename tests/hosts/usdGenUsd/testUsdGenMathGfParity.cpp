// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
// Bitwise parity of the usdGenMath types against Gf 26.08 (step 11.4a/11.4).
// Every operation the engine uses runs on the same input bits through both
// type sets; results compare byte for byte (a NaN matches any NaN: payload
// propagation depends on operand order the compiler may swap). Also the
// traps of the step-11 brief: scalar ops in double (T5), Normalize's eps,
// the empty range and NaN bounds (T6), GetInverse (T7), half rounding (T8).
#include "usdGenMath/math.h"

#include "pxr/base/gf/half.h"
#include "pxr/base/gf/matrix2d.h"
#include "pxr/base/gf/matrix3d.h"
#include "pxr/base/gf/matrix4d.h"
#include "pxr/base/gf/matrix4f.h"
#include "pxr/base/gf/quatd.h"
#include "pxr/base/gf/quatf.h"
#include "pxr/base/gf/quath.h"
#include "pxr/base/gf/range3d.h"
#include "pxr/base/gf/range3f.h"
#include "pxr/base/gf/rotation.h"
#include "pxr/base/gf/vec2d.h"
#include "pxr/base/gf/vec2f.h"
#include "pxr/base/gf/vec2h.h"
#include "pxr/base/gf/vec2i.h"
#include "pxr/base/gf/vec3d.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/base/gf/vec3h.h"
#include "pxr/base/gf/vec3i.h"
#include "pxr/base/gf/vec4d.h"
#include "pxr/base/gf/vec4f.h"
#include "pxr/base/gf/vec4h.h"
#include "pxr/base/gf/vec4i.h"

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

// --- the two type sets -----------------------------------------------------

struct U {
    using V2f = usdGen::UsdGenVec2f; using V3f = usdGen::UsdGenVec3f; using V4f = usdGen::UsdGenVec4f;
    using V2d = usdGen::UsdGenVec2d; using V3d = usdGen::UsdGenVec3d; using V4d = usdGen::UsdGenVec4d;
    using V2i = usdGen::UsdGenVec2i; using V3i = usdGen::UsdGenVec3i; using V4i = usdGen::UsdGenVec4i;
    using V2h = usdGen::UsdGenVec2h; using V3h = usdGen::UsdGenVec3h; using V4h = usdGen::UsdGenVec4h;
    using H = usdGen::UsdGenHalf;
    using M2d = usdGen::UsdGenMatrix2d; using M3d = usdGen::UsdGenMatrix3d;
    using M4d = usdGen::UsdGenMatrix4d; using M4f = usdGen::UsdGenMatrix4f;
    using Qf = usdGen::UsdGenQuatf; using Qd = usdGen::UsdGenQuatd; using Qh = usdGen::UsdGenQuath;
    using R3f = usdGen::UsdGenRange3f; using R3d = usdGen::UsdGenRange3d;
    template <class A, class B> static auto Dot(A const &a, B const &b) { return usdGen::UsdGenDot(a, b); }
    template <class A, class B> static auto Cross(A const &a, B const &b) { return usdGen::UsdGenCross(a, b); }
    static V3d RotateInto(V3d const &a, V3d const &b, V3d const &v) { return usdGen::UsdGenRotateInto(a, b, v); }
    static constexpr double kMinLen = usdGen::kUsdGenMinVectorLength;
};

struct G {
    using V2f = PXR_NS::GfVec2f; using V3f = PXR_NS::GfVec3f; using V4f = PXR_NS::GfVec4f;
    using V2d = PXR_NS::GfVec2d; using V3d = PXR_NS::GfVec3d; using V4d = PXR_NS::GfVec4d;
    using V2i = PXR_NS::GfVec2i; using V3i = PXR_NS::GfVec3i; using V4i = PXR_NS::GfVec4i;
    using V2h = PXR_NS::GfVec2h; using V3h = PXR_NS::GfVec3h; using V4h = PXR_NS::GfVec4h;
    using H = PXR_NS::GfHalf;
    using M2d = PXR_NS::GfMatrix2d; using M3d = PXR_NS::GfMatrix3d;
    using M4d = PXR_NS::GfMatrix4d; using M4f = PXR_NS::GfMatrix4f;
    using Qf = PXR_NS::GfQuatf; using Qd = PXR_NS::GfQuatd; using Qh = PXR_NS::GfQuath;
    using R3f = PXR_NS::GfRange3f; using R3d = PXR_NS::GfRange3d;
    template <class A, class B> static auto Dot(A const &a, B const &b) { return PXR_NS::GfDot(a, b); }
    template <class A, class B> static auto Cross(A const &a, B const &b) { return PXR_NS::GfCross(a, b); }
    static V3d RotateInto(V3d const &a, V3d const &b, V3d const &v) { return PXR_NS::GfRotation(a, b).TransformDir(v); }
    static constexpr double kMinLen = GF_MIN_VECTOR_LENGTH;
};

#define LAYOUT(T) \
    static_assert(sizeof(U::T) == sizeof(G::T) && alignof(U::T) == alignof(G::T) && \
                  std::is_trivially_copyable<U::T>::value, #T " layout");
LAYOUT(V2f) LAYOUT(V3f) LAYOUT(V4f) LAYOUT(V2d) LAYOUT(V3d) LAYOUT(V4d)
LAYOUT(V2i) LAYOUT(V3i) LAYOUT(V4i) LAYOUT(V2h) LAYOUT(V3h) LAYOUT(V4h)
LAYOUT(H) LAYOUT(M2d) LAYOUT(M3d) LAYOUT(M4d) LAYOUT(M4f) LAYOUT(Qf) LAYOUT(Qd) LAYOUT(Qh)
LAYOUT(R3f) LAYOUT(R3d)
#undef LAYOUT

// --- results: scalars with a kind, compared bitwise -------------------------

enum class Kind : uint8_t { F16, F32, F64, I32, Bool };

struct Sink {
    struct Mark { char const *op; size_t at; size_t bytes; Kind kind; };
    std::vector<unsigned char> bytes;
    std::vector<Mark> marks;
    void Raw(char const *op, void const *p, size_t n, Kind k)
    {
        marks.push_back({op, bytes.size(), n, k});
        auto const *c = static_cast<unsigned char const *>(p);
        bytes.insert(bytes.end(), c, c + n);
    }
};

template <class S> constexpr Kind KindOfScalar()
{
    return std::is_same<S, bool>::value ? Kind::Bool
         : std::is_integral<S>::value   ? Kind::I32
         : sizeof(S) == 2               ? Kind::F16
         : sizeof(S) == 4               ? Kind::F32
                                        : Kind::F64;
}
template <class T, class = void> struct ScalarOf { using type = T; };
template <class T> struct ScalarOf<T, std::void_t<typename T::ScalarType>> { using type = typename T::ScalarType; };

template <class T> void Put(Sink &s, char const *op, T const &v)
{
    static_assert(std::is_trivially_copyable<T>::value, "trivial");
    s.Raw(op, &v, sizeof(T), KindOfScalar<typename ScalarOf<T>::type>());
}

bool IsNanBits(unsigned char const *p, Kind k)
{
    switch (k) {
    case Kind::F16: { uint16_t h; std::memcpy(&h, p, 2); return (h & 0x7c00) == 0x7c00 && (h & 0x3ff); }
    case Kind::F32: { float f; std::memcpy(&f, p, 4); return std::isnan(f); }
    case Kind::F64: { double d; std::memcpy(&d, p, 8); return std::isnan(d); }
    default: return false;
    }
}
size_t ScalarBytes(Kind k)
{
    switch (k) {
    case Kind::F16: return 2;
    case Kind::F32: case Kind::I32: return 4;
    case Kind::F64: return 8;
    case Kind::Bool: return 1;
    }
    return 1;
}

int g_failures = 0;
size_t g_checks = 0;
std::map<std::string, size_t> g_failuresByOp;

void Fail(char const *what, size_t index, std::string const &detail)
{
    ++g_failuresByOp[what];
    if (++g_failures <= 40) std::printf("FAIL %s [case %zu] %s\n", what, index, detail.c_str());
}

std::string Hex(unsigned char const *p, size_t n)
{
    std::string out;
    char buf[4];
    for (size_t i = 0; i < n; ++i) { std::snprintf(buf, sizeof(buf), "%02x", p[i]); out += buf; }
    return out;
}

void Compare(char const *suite, size_t caseIndex, Sink const &u, Sink const &g)
{
    if (u.marks.size() != g.marks.size()) {
        Fail(suite, caseIndex, "op count differs");
        return;
    }
    for (size_t m = 0; m < u.marks.size(); ++m) {
        Sink::Mark const &a = u.marks[m];
        Sink::Mark const &b = g.marks[m];
        ++g_checks;
        if (a.bytes != b.bytes || std::strcmp(a.op, b.op) != 0) {
            Fail(a.op, caseIndex, "shape differs");
            continue;
        }
        size_t const w = ScalarBytes(a.kind);
        unsigned char const *pa = u.bytes.data() + a.at;
        unsigned char const *pb = g.bytes.data() + b.at;
        bool same = true;
        for (size_t i = 0; i + w <= a.bytes && same; i += w) {
            if (std::memcmp(pa + i, pb + i, w) == 0) continue;
            if (IsNanBits(pa + i, a.kind) && IsNanBits(pb + i, a.kind)) continue;
            same = false;
        }
        if (!same) Fail(a.op, caseIndex, "usdGen " + Hex(pa, a.bytes) + " gf " + Hex(pb, b.bytes));
    }
}

// --- inputs ------------------------------------------------------------------

struct Rng {
    uint64_t s;
    uint64_t Next() { uint64_t z = (s += 0x9e3779b97f4a7c15ull); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull; z = (z ^ (z >> 27)) * 0x94d049bb133111ebull; return z ^ (z >> 31); }
    double Unit() { return double(Next() >> 11) * (1.0 / 9007199254740992.0); }
};

double const kSpecialD[] = {0.0, -0.0, 1.0, -1.0, 0.5, 2.0, 1e-10, -1e-10, 1e-20, 1e-300, 4.9e-324,
                            1e300, DBL_MAX, -DBL_MAX, DBL_MIN, std::numeric_limits<double>::infinity(),
                            -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN(),
                            3.0e38, -3.0e38, 1.17549435e-38, 1.4e-45, 65504.0, 6.1e-5, 1e-7};

// One input double: mostly well-conditioned, sometimes special or raw bits.
double Draw(Rng &r)
{
    uint64_t const sel = r.Next() % 16;
    if (sel == 0) return kSpecialD[r.Next() % (sizeof(kSpecialD) / sizeof(kSpecialD[0]))];
    if (sel == 1) { uint64_t b = r.Next(); double d; std::memcpy(&d, &b, 8); return d; }
    if (sel == 2) { uint32_t b = uint32_t(r.Next()); float f; std::memcpy(&f, &b, 4); return f; }
    if (sel == 3) return (r.Unit() - 0.5) * 1e-6;
    if (sel == 4) return std::floor((r.Unit() - 0.5) * 16.0);
    return (r.Unit() - 0.5) * 200.0;
}

struct Input {
    double d[48];
    int i[12];
    double s;        // scalar operand
    int si;          // int scalar operand (nonzero)
};

Input MakeInput(Rng &r)
{
    Input in;
    for (double &x : in.d) x = Draw(r);
    for (int &x : in.i) x = int(int64_t(r.Next() % 2001) - 1000);
    in.s = Draw(r);
    in.si = int(r.Next() % 19) - 9;
    if (in.si == 0) in.si = 3;
    return in;
}

template <class V> V MakeVec(double const *d)
{
    using S = typename V::ScalarType;
    V v;
    for (size_t k = 0; k < V::dimension; ++k) v[k] = S(float(d[k]));
    return v;
}
template <class V> V MakeVecD(double const *d)
{
    using S = typename V::ScalarType;
    V v;
    for (size_t k = 0; k < V::dimension; ++k) v[k] = S(d[k]);
    return v;
}
template <class V> V MakeVecI(int const *d)
{
    using S = typename V::ScalarType;
    V v;
    for (size_t k = 0; k < V::dimension; ++k) v[k] = S(d[k]);
    return v;
}

// --- vector operations (every one the engine uses, plus the Gf surface) ------

template <class K, class V> void FloatVecOps(Sink &s, char const *tag, V a, V b, double sc)
{
    using S = typename V::ScalarType;
    Put(s, tag, a);
    Put(s, "neg", -a);
    Put(s, "add", a + b);
    Put(s, "sub", a - b);
    { V c = a; c += b; Put(s, "addeq", c); }
    { V c = a; c -= b; Put(s, "subeq", c); }
    Put(s, "muls", a * sc);
    Put(s, "smul", sc * a);
    Put(s, "mulf", a * float(sc));
    Put(s, "muli", a * 3);
    { V c = a; c *= sc; Put(s, "muleq", c); }
    Put(s, "divs", a / sc);
    Put(s, "divf", a / float(sc));
    { V c = a; c /= sc; Put(s, "diveq", c); }
    Put(s, "dot", S(a * b));
    Put(s, "Dot", S(K::Dot(a, b)));
    Put(s, "lensq", S(a.GetLengthSq()));
    Put(s, "len", S(a.GetLength()));
    { V c = a; S l = c.Normalize(); Put(s, "norm.len", l); Put(s, "norm", c); }
    { V c = a; S l = c.Normalize(S(1e-3)); Put(s, "norm3.len", l); Put(s, "norm3", c); }
    Put(s, "normalized", a.GetNormalized());
    Put(s, "normalized3", a.GetNormalized(S(0.5)));
    Put(s, "proj", a.GetProjection(b));
    Put(s, "compl", a.GetComplement(b));
    Put(s, "eq", bool(a == b));
    Put(s, "eqself", bool(a == a));
    Put(s, "ne", bool(a != b));
    Put(s, "bcast", V(S(sc)));
    Put(s, "ptr", V(a.data()));
    Put(s, "getarray", V(a.GetArray()));
    Put(s, "index0", S(a[0]));
}

template <class K, class V> void IntVecOps(Sink &s, V a, V b, double sc, int si)
{
    Put(s, "i", a);
    Put(s, "i.neg", -a);
    Put(s, "i.add", a + b);
    Put(s, "i.sub", a - b);
    { V c = a; c += b; Put(s, "i.addeq", c); }
    { V c = a; c -= b; Put(s, "i.subeq", c); }
    double const small = std::fmod(sc, 4.0);   // keep int conversions defined
    double const safe = std::isfinite(small) ? small : 0.5;
    Put(s, "i.muls", a * safe);
    Put(s, "i.smul", safe * a);
    { V c = a; c *= safe; Put(s, "i.muleq", c); }
    Put(s, "i.divi", a / si);
    { V c = a; c /= si; Put(s, "i.diveq", c); }
    Put(s, "i.dot", int(a * b));
    Put(s, "i.Dot", int(K::Dot(a, b)));
    Put(s, "i.lensq", int(a.GetLengthSq()));
    Put(s, "i.eq", bool(a == b));
    Put(s, "i.eqself", bool(a == a));
    Put(s, "i.bcast", V(si));
    Put(s, "i.ptr", V(a.data()));
}

template <class K> void VecSuite(Sink &s, Input const &in)
{
    using V2f = typename K::V2f; using V3f = typename K::V3f; using V4f = typename K::V4f;
    using V2d = typename K::V2d; using V3d = typename K::V3d; using V4d = typename K::V4d;
    using V2i = typename K::V2i; using V3i = typename K::V3i; using V4i = typename K::V4i;
    using V2h = typename K::V2h; using V3h = typename K::V3h; using V4h = typename K::V4h;
    using H = typename K::H;

    FloatVecOps<K>(s, "v2f", MakeVec<V2f>(in.d), MakeVec<V2f>(in.d + 4), in.s);
    FloatVecOps<K>(s, "v3f", MakeVec<V3f>(in.d), MakeVec<V3f>(in.d + 4), in.s);
    FloatVecOps<K>(s, "v4f", MakeVec<V4f>(in.d), MakeVec<V4f>(in.d + 4), in.s);
    FloatVecOps<K>(s, "v2d", MakeVecD<V2d>(in.d + 8), MakeVecD<V2d>(in.d + 12), in.s);
    FloatVecOps<K>(s, "v3d", MakeVecD<V3d>(in.d + 8), MakeVecD<V3d>(in.d + 12), in.s);
    FloatVecOps<K>(s, "v4d", MakeVecD<V4d>(in.d + 8), MakeVecD<V4d>(in.d + 12), in.s);
    FloatVecOps<K>(s, "v2h", MakeVec<V2h>(in.d + 16), MakeVec<V2h>(in.d + 20), in.s);
    FloatVecOps<K>(s, "v3h", MakeVec<V3h>(in.d + 16), MakeVec<V3h>(in.d + 20), in.s);
    FloatVecOps<K>(s, "v4h", MakeVec<V4h>(in.d + 16), MakeVec<V4h>(in.d + 20), in.s);
    IntVecOps<K>(s, MakeVecI<V2i>(in.i), MakeVecI<V2i>(in.i + 4), in.s, in.si);
    IntVecOps<K>(s, MakeVecI<V3i>(in.i), MakeVecI<V3i>(in.i + 4), in.s, in.si);
    IntVecOps<K>(s, MakeVecI<V4i>(in.i), MakeVecI<V4i>(in.i + 4), in.s, in.si);

    V3f const f = MakeVec<V3f>(in.d), f2 = MakeVec<V3f>(in.d + 4);
    V3d const d = MakeVecD<V3d>(in.d + 8), d2 = MakeVecD<V3d>(in.d + 12);
    V3h const h = MakeVec<V3h>(in.d + 16);
    V3i const iv = MakeVecI<V3i>(in.i);

    // Cross products.
    Put(s, "cross3f", K::Cross(f, f2));
    Put(s, "cross3d", K::Cross(d, d2));
    Put(s, "cross3h", K::Cross(h, MakeVec<V3h>(in.d + 20)));
    // Mixed overload resolution: float promotes to double, never the reverse.
    Put(s, "mix.dot.fd", double(K::Dot(f, d)));
    Put(s, "mix.dot.df", double(K::Dot(d, f)));
    Put(s, "mix.cross.fd", K::Cross(f, d));
    Put(s, "mix.add.fd", f + d);
    Put(s, "mix.sub.df", d - f);
    Put(s, "mix.eq.fd", bool(f == d));
    Put(s, "mix.eq.df", bool(d == V3d(f)));
    Put(s, "mix.eq.ff", bool(f == V3f(d)));
    // Conversions (Gf's table).
    Put(s, "conv.f2d", V3d(f));
    Put(s, "conv.d2f", V3f(d));
    Put(s, "conv.h2f", V3f(h));
    Put(s, "conv.h2d", V3d(h));
    Put(s, "conv.f2h", V3h(f));
    Put(s, "conv.d2h", V3h(d));
    Put(s, "conv.i2f", V3f(iv));
    Put(s, "conv.i2d", V3d(iv));
    Put(s, "conv.i2h", V3h(iv));
    Put(s, "conv.2f2d", V2d(MakeVec<V2f>(in.d)));
    Put(s, "conv.2d2f", V2f(MakeVecD<V2d>(in.d + 8)));
    Put(s, "conv.4f2d", V4d(MakeVec<V4f>(in.d)));
    Put(s, "conv.4d2f", V4f(MakeVecD<V4d>(in.d + 8)));
    { V3d imp = f; Put(s, "conv.implicit.f2d", imp); }
    { V3f imp = h; Put(s, "conv.implicit.h2f", imp); }
    { V3f imp = iv; Put(s, "conv.implicit.i2f", imp); }
    // Constructors.
    Put(s, "ctor3f", V3f(float(in.d[0]), float(in.d[1]), float(in.d[2])));
    Put(s, "ctor3f.d", V3f(in.d[0], in.d[1], in.d[2]));
    Put(s, "ctor2f", V2f(float(in.d[0]), float(in.d[1])));
    Put(s, "ctor4d", V4d(in.d[0], in.d[1], in.d[2], in.d[3]));
    Put(s, "ctor3h", V3h(H(float(in.d[16])), H(float(in.d[17])), H(float(in.d[18]))));
    Put(s, "ctor3i", V3i(in.i[0], in.i[1], in.i[2]));
    { float const p[3] = {float(in.d[0]), float(in.d[1]), float(in.d[2])}; Put(s, "ctor3f.ptr", V3f(p)); }
    { double const p[3] = {in.d[0], in.d[1], in.d[2]}; Put(s, "ctor3d.ptr", V3d(p)); }
    Put(s, "valueinit", V3f{});
    // Scalar types of results.
    Put(s, "len3f.type", sizeof(f.GetLength()) == 4);
    Put(s, "len3d.type", sizeof(d.GetLength()) == 8);
    Put(s, "dot3h.type", sizeof(h * h) == 2);
    Put(s, "minlen", K::kMinLen);
}

// --- half --------------------------------------------------------------------

template <class K> void HalfSuite(Sink &s, Input const &in)
{
    using H = typename K::H;
    float const a = float(in.d[24]), b = float(in.d[25]);
    H const ha(a), hb(b);
    Put(s, "h.ctor", ha);
    Put(s, "h.float", float(ha));
    Put(s, "h.neg", H(-ha));
    Put(s, "h.add", float(ha + hb));
    Put(s, "h.mul", float(ha * hb));
    Put(s, "h.div", float(ha / hb));
    Put(s, "h.muld", double(ha * in.s));
    { H c = ha; c += hb; Put(s, "h.addeq", c); }
    { H c = ha; c -= hb; Put(s, "h.subeq", c); }
    { H c = ha; c *= hb; Put(s, "h.muleq", c); }
    { H c = ha; c /= hb; Put(s, "h.diveq", c); }
    { H c = ha; c += b; Put(s, "h.addeqf", c); }
    { H c = ha; c *= float(in.s); Put(s, "h.muleqf", c); }
    { H c = ha; c = b; Put(s, "h.assignf", c); }
    Put(s, "h.eq", bool(ha == hb));
    Put(s, "h.lt", bool(ha < hb));
    Put(s, "h.int", H(in.i[0]));
    Put(s, "h.double", H(float(in.d[26])));
    Put(s, "h.bits", uint32_t(ha.bits()));
}

// --- matrices ----------------------------------------------------------------

template <class M> M MakeMat4(double const *d)
{
    return M(d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7],
             d[8], d[9], d[10], d[11], d[12], d[13], d[14], d[15]);
}

template <class K> void MatrixOps(Sink &s, typename K::M4d const &m, typename K::M4d const &n,
                                  double const *v, double sc)
{
    using M4d = typename K::M4d; using V3d = typename K::V3d; using V4d = typename K::V4d;
    using V3f = typename K::V3f;
    Put(s, "m", m);
    { double det = 0; M4d inv = m.GetInverse(&det); Put(s, "m.inv", inv); Put(s, "m.inv.det", det); }
    { double det = 0; M4d inv = m.GetInverse(&det, 1e-9); Put(s, "m.inv.eps", inv); Put(s, "m.inv.eps.det", det); }
    Put(s, "m.inv.nodet", m.GetInverse());
    Put(s, "m.det", m.GetDeterminant());
    Put(s, "m.transpose", m.GetTranspose());
    Put(s, "m.mul", m * n);
    { M4d c = m; c *= n; Put(s, "m.muleq", c); }
    Put(s, "m.muls", m * sc);
    Put(s, "m.smul", sc * m);
    { M4d c = m; c *= sc; Put(s, "m.muleqs", c); }
    Put(s, "m.div", m / n);
    V3d const p(v[0], v[1], v[2]);
    V4d const p4(v[0], v[1], v[2], v[3]);
    Put(s, "m.xf", m.Transform(p));
    Put(s, "m.xfdir", m.TransformDir(p));
    Put(s, "m.xfaff", m.TransformAffine(p));
    V3f const pf = V3f(float(v[0]), float(v[1]), float(v[2]));
    Put(s, "m.xf.f", m.Transform(pf));            // Vec3f converts to Vec3d (no Vec3f overload)
    Put(s, "m.xfdir.f", m.TransformDir(pf));
    Put(s, "m.xfaff.f", m.TransformAffine(pf));
    Put(s, "m.mv4", m * p4);
    Put(s, "m.v4m", p4 * m);
    Put(s, "m.row1", m.GetRow(1));
    Put(s, "m.row3", m.GetRow3(2));
    { M4d c = m; c.SetRow(2, p4); Put(s, "m.setrow", c); }
    { M4d c = m; c.SetIdentity(); Put(s, "m.setidentity", c); }
    { M4d c = m; c.SetDiagonal(sc); Put(s, "m.setdiag", c); }
    { M4d c = m; c.SetTranslate(p); Put(s, "m.settranslate", c); }
    { M4d c = m; c.SetTranslateOnly(p); Put(s, "m.settranslateonly", c); }
    { M4d c = m; c.SetScale(p); Put(s, "m.setscale3", c); }
    { M4d c = m; c.SetScale(sc); Put(s, "m.setscale", c); }
    Put(s, "m.diag", M4d(sc));
    Put(s, "m.identity", M4d(1.0));
    Put(s, "m.eq", bool(m == n));
    Put(s, "m.eqself", bool(m == m));
    Put(s, "m.ne", bool(m != n));
    Put(s, "m.elem", m[2][1]);
    Put(s, "m.data", m.data()[7]);
    Put(s, "m.getarray", m.GetArray()[11]);
    { double raw[4][4]; for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) raw[r][c] = v[(r * 4 + c) % 4] * (r + 1); Put(s, "m.ctorarr", M4d(raw)); }
}

template <class K> void MatrixSuite(Sink &s, Input const &in, Rng &r)
{
    using M4d = typename K::M4d; using M4f = typename K::M4f;
    using M2d = typename K::M2d; using M3d = typename K::M3d;
    M4d const m = MakeMat4<M4d>(in.d);
    M4d const n = MakeMat4<M4d>(in.d + 16);
    MatrixOps<K>(s, m, n, in.d + 32, in.s);
    (void)r;
    Put(s, "m4f.from4d", M4f(m));
    Put(s, "m4d.from4f", M4d(M4f(m)));
    Put(s, "m2d.diag", M2d(in.s));
    Put(s, "m3d.diag", M3d(in.s));
    Put(s, "m2d.ident", M2d(1.0));
    Put(s, "m3d.ident", M3d(1.0));
    Put(s, "m2d.eq", bool(M2d(in.s) == M2d(1.0)));
    Put(s, "m3d.eq", bool(M3d(in.s) == M3d(in.s)));
}

// Well-conditioned, near-singular, exactly singular and non-affine matrices.
void ConditionedMatrix(Rng &r, int mode, double out[16])
{
    for (int k = 0; k < 16; ++k) out[k] = (r.Unit() - 0.5) * 4.0;
    if (mode == 1) {           // near-singular: row 3 = row 0 + tiny
        double const tiny = std::pow(10.0, -12.0 - 288.0 * r.Unit());
        for (int c = 0; c < 4; ++c) out[12 + c] = out[c] + tiny * (r.Unit() - 0.5);
    } else if (mode == 2) {    // exactly singular
        for (int c = 0; c < 4; ++c) out[8 + c] = out[c] * 2.0;
    } else if (mode == 3) {    // affine
        out[3] = out[7] = out[11] = 0.0;
        out[15] = 1.0;
    } else if (mode == 4) {    // scale/rotate/translate magnitude mix
        for (int k = 0; k < 16; ++k) out[k] *= std::pow(10.0, double(int(r.Next() % 13)) - 6.0);
    }
}

template <class K> void InverseSuite(Sink &s, double const m16[16], double const v[4])
{
    using M4d = typename K::M4d; using V3d = typename K::V3d;
    M4d const m = MakeMat4<M4d>(m16);
    double det = 0;
    Put(s, "inv", m.GetInverse(&det));
    Put(s, "inv.det", det);
    Put(s, "inv.eps", m.GetInverse(nullptr, 1e-12));
    Put(s, "det", m.GetDeterminant());
    Put(s, "inv.xf", m.GetInverse().Transform(V3d(v[0], v[1], v[2])));
    Put(s, "mul.inv", m * m.GetInverse());
}

// --- ranges ------------------------------------------------------------------

template <class K, class R, class V> void RangeOps(Sink &s, char const *tag, V const *pts, size_t n, V lo, V hi)
{
    R r;
    Put(s, tag, r);
    Put(s, "r.emptyinit", bool(r.IsEmpty()));
    for (size_t k = 0; k < n; ++k) {
        r.UnionWith(pts[k]);
        Put(s, "r.union.pt", r);
    }
    Put(s, "r.empty", bool(r.IsEmpty()));
    Put(s, "r.min", r.GetMin());
    Put(s, "r.max", r.GetMax());
    Put(s, "r.size", r.GetSize());
    Put(s, "r.contains", bool(r.Contains(pts[0])));
    Put(s, "r.contains.lo", bool(r.Contains(lo)));
    R b(lo, hi);
    Put(s, "r.ctor", b);
    Put(s, "r.ctor.empty", bool(b.IsEmpty()));
    { R c = r; c.UnionWith(b); Put(s, "r.union.range", c); }
    { R c = b; c.ExtendBy(pts[1]); Put(s, "r.extend.pt", c); }
    { R c = b; c.ExtendBy(r); Put(s, "r.extend.range", c); }
    { R c = b; c.SetEmpty(); Put(s, "r.setempty", c); }
    { R c; c.UnionWith(R()); Put(s, "r.union.empty", c); }
    Put(s, "r.mid", b.GetMidpoint());
    Put(s, "r.outside", bool(r.IsOutside(b)));
    Put(s, "r.getunion", R::GetUnion(r, b));
    Put(s, "r.getintersection", R::GetIntersection(r, b));
    { R c = b; c.IntersectWith(r); Put(s, "r.intersectwith", c); }
    { R c = b; c *= -2.5; Put(s, "r.muleq", c); }
    Put(s, "r.eq", bool(r == b));
    Put(s, "r.eqself", bool(b == b));
}

template <class K> void RangeSuite(Sink &s, Input const &in)
{
    using V3f = typename K::V3f; using V3d = typename K::V3d;
    V3f pf[6]; V3d pd[6];
    for (int k = 0; k < 6; ++k) { pf[k] = MakeVec<V3f>(in.d + 3 * k); pd[k] = MakeVecD<V3d>(in.d + 3 * k); }
    RangeOps<K, typename K::R3f>(s, "r3f", pf, 6, MakeVec<V3f>(in.d + 30), MakeVec<V3f>(in.d + 33));
    RangeOps<K, typename K::R3d>(s, "r3d", pd, 6, MakeVecD<V3d>(in.d + 30), MakeVecD<V3d>(in.d + 33));
}

// --- quaternions -------------------------------------------------------------

template <class K> void QuatSuite(Sink &s, Input const &in)
{
    using Qf = typename K::Qf; using Qd = typename K::Qd; using Qh = typename K::Qh;
    using V3f = typename K::V3f; using V3d = typename K::V3d; using V3h = typename K::V3h;
    using H = typename K::H;
    Qd const qd(in.d[0], V3d(in.d[1], in.d[2], in.d[3]));
    Qf const qf(float(in.d[4]), V3f(float(in.d[5]), float(in.d[6]), float(in.d[7])));
    Qh const qh(H(float(in.d[8])), V3h(H(float(in.d[9])), H(float(in.d[10])), H(float(in.d[11]))));
    Put(s, "qd", qd); Put(s, "qf", qf); Put(s, "qh", qh);
    Put(s, "qd.real", qd.GetReal()); Put(s, "qd.imag", qd.GetImaginary());
    Put(s, "qf.real", qf.GetReal()); Put(s, "qf.imag", qf.GetImaginary());
    Put(s, "qh.real", qh.GetReal()); Put(s, "qh.imag", qh.GetImaginary());
    Put(s, "qd.ident", Qd(1.0)); Put(s, "qf.ident", Qf(1.0f)); Put(s, "qh.ident", Qh(H(1.0f)));
    Put(s, "qd.eq", bool(qd == qd)); Put(s, "qf.eq", bool(qf == Qf(1.0f)));
    Put(s, "qd.len", qd.GetLength()); Put(s, "qf.len", qf.GetLength());
    Put(s, "qd.normalized", qd.GetNormalized()); Put(s, "qf.normalized", qf.GetNormalized());
    Put(s, "qd.conj", qd.GetConjugate()); Put(s, "qd.inv", qd.GetInverse()); Put(s, "qf.inv", qf.GetInverse());
    Put(s, "qd.add", qd + qd.GetConjugate()); Put(s, "qd.muls", qd * 1.5); Put(s, "qf.divs", qf / 3.0f);
    Put(s, "qd.neg", -qd);
    Put(s, "qd.dot", double(K::Dot(qd, qd.GetConjugate())));
    Put(s, "qd.fromf", Qd(qf));
}

// --- rotation (curveWrap) ----------------------------------------------------

template <class K> void RotateSuite(Sink &s, Input const &in, Rng &r)
{
    using V3d = typename K::V3d;
    V3d const a = MakeVecD<V3d>(in.d), b = MakeVecD<V3d>(in.d + 3), v = MakeVecD<V3d>(in.d + 6);
    Put(s, "rot", K::RotateInto(a, b, v));
    Put(s, "rot.same", K::RotateInto(a, a, v));
    Put(s, "rot.opposite", K::RotateInto(a, -a, v));
    Put(s, "rot.xaxis", K::RotateInto(V3d(1, 0, 0), -V3d(1, 0, 0), v));
    Put(s, "rot.near", K::RotateInto(a, a + V3d(1e-9, -1e-9, 1e-9), v));
    Put(s, "rot.zero", K::RotateInto(V3d(0, 0, 0), b, v));
    (void)r;
}

template <class F> void ForEachCase(char const *suite, size_t count, uint64_t seed, F f)
{
    Rng r{seed};
    for (size_t c = 0; c < count; ++c) {
        Input const in = MakeInput(r);
        Rng ru{r.Next()};
        Rng rg = ru;
        Sink su, sg;
        f(su, sg, in, ru, rg);
        Compare(suite, c, su, sg);
    }
}

// --- exhaustive half conversions (T8) ----------------------------------------

void HalfExhaustive()
{
    // All 65,536 half -> float.
    for (uint32_t b = 0; b < 65536; ++b) {
        U::H hu; hu.setBits(uint16_t(b));
        G::H hg; hg.setBits(uint16_t(b));
        float const fu = hu, fg = hg;
        ++g_checks;
        if (std::memcmp(&fu, &fg, 4) != 0) Fail("half->float", b, "");
    }
    // All 2^32 float -> half, split across threads.
    unsigned const nt = std::max(1u, std::thread::hardware_concurrency());
    std::atomic<uint64_t> bad{0};
    std::atomic<uint64_t> firstBad{~0ull};
    std::vector<std::thread> threads;
    uint64_t const total = 1ull << 32;
    for (unsigned t = 0; t < nt; ++t) {
        threads.emplace_back([&, t] {
            uint64_t const lo = total * t / nt, hi = total * (t + 1) / nt;
            uint64_t localBad = 0;
            for (uint64_t b = lo; b < hi; ++b) {
                uint32_t const bits = uint32_t(b);
                float f;
                std::memcpy(&f, &bits, 4);
                U::H const hu(f);
                G::H const hg(f);
                if (hu.bits() != hg.bits()) {
                    if (!localBad) {
                        uint64_t expected = ~0ull;
                        firstBad.compare_exchange_strong(expected, b);
                    }
                    ++localBad;
                }
            }
            bad += localBad;
        });
    }
    for (std::thread &t : threads) t.join();
    g_checks += total;
    if (bad) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "%llu mismatches, first float bits 0x%08llx",
                      (unsigned long long)bad.load(), (unsigned long long)firstBad.load());
        Fail("float->half", 0, buf);
    }
}

}  // namespace

int main()
{
    ForEachCase("vec", 20000, 1, [](Sink &su, Sink &sg, Input const &in, Rng &, Rng &) {
        VecSuite<U>(su, in);
        VecSuite<G>(sg, in);
    });
    ForEachCase("half", 20000, 2, [](Sink &su, Sink &sg, Input const &in, Rng &, Rng &) {
        HalfSuite<U>(su, in);
        HalfSuite<G>(sg, in);
    });
    ForEachCase("matrix", 20000, 3, [](Sink &su, Sink &sg, Input const &in, Rng &ru, Rng &rg) {
        MatrixSuite<U>(su, in, ru);
        MatrixSuite<G>(sg, in, rg);
    });
    ForEachCase("range", 20000, 4, [](Sink &su, Sink &sg, Input const &in, Rng &, Rng &) {
        RangeSuite<U>(su, in);
        RangeSuite<G>(sg, in);
    });
    ForEachCase("quat", 5000, 5, [](Sink &su, Sink &sg, Input const &in, Rng &, Rng &) {
        QuatSuite<U>(su, in);
        QuatSuite<G>(sg, in);
    });
    ForEachCase("rotate", 20000, 6, [](Sink &su, Sink &sg, Input const &in, Rng &ru, Rng &rg) {
        RotateSuite<U>(su, in, ru);
        RotateSuite<G>(sg, in, rg);
    });
    // T7: 10^6 inverses over conditioned, near-singular, singular, non-affine.
    {
        Rng r{7};
        for (size_t c = 0; c < 1000000; ++c) {
            double m[16], v[4];
            ConditionedMatrix(r, int(c % 5), m);
            for (double &x : v) x = (r.Unit() - 0.5) * 10.0;
            Sink su, sg;
            InverseSuite<U>(su, m, v);
            InverseSuite<G>(sg, m, v);
            Compare("inverse", c, su, sg);
        }
    }
    HalfExhaustive();
    for (auto const &f : g_failuresByOp) std::printf("  %s: %zu failures\n", f.first.c_str(), f.second);
    std::printf("testUsdGenMathGfParity: %zu checks, %d failures\n", g_checks, g_failures);
    return g_failures ? 1 : 0;
}
