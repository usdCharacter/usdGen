// usdGen — small helpers shared by the CPU operators that read whole
// buffers at capture (Clump, GuideInterpolate).
#ifndef USDGEN_OP_UTIL_H
#define USDGEN_OP_UTIL_H

#include "usdGen/curveBuffer.h"
#include "usdGenSurface/surfaceInput.h"
#include "usdGen/api/graphSpec.h"
#include "usdGen/op.h"
#include "usdGenCurves/types/curveSet.h"

#include "usdGenMath/math.h"
#include "usdGenMath/math.h"
#include "usdGen/api/value.h"
#include "usdGen/api/token.h"
#include "usdGen/api/span.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>


namespace usdGen {
namespace opUtil {

/// FNV-1a over capture-class inputs.
struct Digest
{
    uint64_t h = 1469598103934665603ULL;
    void Mix(uint64_t v) { h ^= v; h *= 0x100000001b3ULL; }
    void Mix(double v)
    {
        uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        Mix(bits);
    }
    void Mix(UsdGenToken const &t) { for (char const *p = t.GetText(); *p; ++p) Mix(uint64_t(uint8_t(*p))); }
    void Mix(UsdGenValue const &v) { Mix(uint64_t(v.IsEmpty() ? 0 : v.GetHash())); }
    /// FNV-1a over a byte range, one round per 8-byte word. The multiply
    /// chain is the bottleneck, so groups of four words run as four
    /// independent lanes (distinct seeds, folded back in order). Faster
    /// than a serial word loop; the epoch values differ from serial
    /// mixing, but epochs are only compared for equality within a process.
    void MixBytes(void const *data, size_t bytes)
    {
        auto const *p = static_cast<unsigned char const *>(data);
        size_t words = bytes / 8;
        if (words >= 4) {
            uint64_t l0 = 1469598103934665603ULL;
            uint64_t l1 = l0 ^ 0x9e3779b97f4a7c15ULL;
            uint64_t l2 = l0 ^ 0xbf58476d1ce4e5b9ULL;
            uint64_t l3 = l0 ^ 0x94d049bb133111ebULL;
            size_t const groups = words / 4;
            for (size_t g = 0; g < groups; ++g) {
                uint64_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
                std::memcpy(&w0, p + g * 32, sizeof(w0));
                std::memcpy(&w1, p + g * 32 + 8, sizeof(w1));
                std::memcpy(&w2, p + g * 32 + 16, sizeof(w2));
                std::memcpy(&w3, p + g * 32 + 24, sizeof(w3));
                l0 ^= w0; l0 *= 0x100000001b3ULL;
                l1 ^= w1; l1 *= 0x100000001b3ULL;
                l2 ^= w2; l2 *= 0x100000001b3ULL;
                l3 ^= w3; l3 *= 0x100000001b3ULL;
            }
            Mix(l0);
            Mix(l1);
            Mix(l2);
            Mix(l3);
            size_t const done = groups * 32;
            p += done;
            bytes -= done;
            words -= groups * 4;
        }
        while (words-- > 0) {
            uint64_t word = 0;
            std::memcpy(&word, p, sizeof(word));
            Mix(word);
            p += 8;
            bytes -= 8;
        }
        for (size_t i = 0; i < bytes; ++i) Mix(uint64_t(p[i]));
    }
    UsdGenEpoch Epoch(uint64_t salt) const { return {h, h ^ salt}; }
};

/// Memoized MixBytes for a rest/topology array that is usually static across
/// poses. The key is the shared buffer's identity (pointer + size), not its
/// content: the cache holds a VtArray reference, and VtArray is
/// copy-on-write, so any in-place edit detaches the writer to a new buffer
/// and the key misses. A hit therefore proves the bytes are unchanged, and a
/// miss rehashes exactly as before, so the epochs are unchanged. For use from
/// const CaptureDigest (mutable member); digests run per node, never
/// concurrently on one op.
template <class T>
class ContentDigestCache
{
public:
    /// 'owner' is the input record's set the values live in (an input
    /// view's 'data'). The keyed set is held, so its memory cannot be reused
    /// and a hit (same address and size) proves the bytes unchanged; another
    /// set sharing the column hits too. Values without an owner (a payload
    /// view) are never keyed.
    uint64_t Digest(UsdGenRef<const UsdGenElementSet> const &owner, UsdGenSpan<const T> values)
    {
        if (valid_ && owner_ && values.size() == size_ &&
            (values.empty() || values.cdata() == data_))
            return digest_;
        opUtil::Digest d;
        d.MixBytes(values.cdata(), values.size() * sizeof(T));
        owner_ = owner;
        data_ = values.cdata();
        size_ = values.size();
        digest_ = d.h;
        valid_ = bool(owner);
        return digest_;
    }

private:
    UsdGenRef<const UsdGenElementSet> owner_;
    T const *data_ = nullptr;
    size_t size_ = 0;
    uint64_t digest_ = 0;
    bool valid_ = false;
};

/// ContentDigestCache for a column of an immutable element set. The key is
/// the column's address and size under the same owning set; the cache holds
/// that set, so the column cannot be freed and reused while keyed, and a hit
/// proves the bytes unchanged. A miss rehashes exactly as MixBytes. Holding
/// the set is a refcount: no allocation.
class SetColumnDigestCache
{
public:
    uint64_t Digest(UsdGenRef<const UsdGenElementSet> const &owner, void const *data, size_t bytes)
    {
        if (valid_ && owner_.Get() == owner.Get() && data_ == data && bytes_ == bytes)
            return digest_;
        opUtil::Digest d;
        d.MixBytes(data, bytes);
        owner_ = owner;
        data_ = data;
        bytes_ = bytes;
        digest_ = d.h;
        valid_ = true;
        return digest_;
    }

private:
    UsdGenRef<const UsdGenElementSet> owner_;
    void const *data_ = nullptr;
    size_t bytes_ = 0;
    uint64_t digest_ = 0;
    bool valid_ = false;
};

/// Curve spans [spans[c], spans[c+1]) of a uniform or ragged buffer.
inline bool CurveSpans(UsdGenCurveBuffer const &buffer, std::vector<uint32_t> *spans,
                       std::string *error)
{
    return detail::_UsdGenCurveSpans(buffer, spans, error);
}

inline bool CurveSpans(UsdGenCurveSetView const &curves, std::vector<uint32_t> *spans,
                       std::string *error)
{
    return detail::_UsdGenCurveSpans(curves, spans, error);
}

inline UsdGenVec3f Point(UsdGenCurveBuffer const &b, size_t cv)
{
    return UsdGenVec3f(b.px[cv], b.py[cv], b.pz[cv]);
}

inline UsdGenVec3f Point(UsdGenCurveSetView const &b, size_t cv)
{
    return UsdGenVec3f(b.px[cv], b.py[cv], b.pz[cv]);
}

/// The rest position of CV `cv` when the buffer carries a complete rest
/// plane, else its current position.
inline UsdGenVec3f RestPoint(UsdGenCurveBuffer const &b, size_t cv)
{
    return b.rest.size() == b.totalCvs ? b.rest[cv] : Point(b, cv);
}

inline UsdGenVec3f RestPoint(UsdGenCurveSetView const &b, size_t cv)
{
    return b.rest.size() == b.totalCvs ? b.rest[cv] : Point(b, cv);
}

inline UsdGenVec3f Normalized(UsdGenVec3f v, UsdGenVec3f fallback = UsdGenVec3f(0, 1, 0))
{
    float const length = v.GetLength();
    return length > 1e-12f ? v / length : fallback;
}

/// Rotates `v` by the rotation that takes unit `from` onto unit `to`, scaled
/// to `amount` of its angle (0 = identity, 1 = the full minimal rotation).
inline UsdGenVec3f RotateOnto(UsdGenVec3f const &v, UsdGenVec3f const &from, UsdGenVec3f const &to,
                          float amount)
{
    if (amount <= 0.0f) return v;
    UsdGenVec3d const a(from), b(to);
    UsdGenVec3d axis = UsdGenCross(a, b);
    double const s = axis.GetLength();
    double const c = UsdGenDot(a, b);
    double angle = std::atan2(s, c) * double(amount);
    if (s < 1e-9) {
        if (c > 0.0) return v;                 // already aligned
        // Opposite: any axis perpendicular to `from`.
        axis = std::fabs(a[0]) < 0.9 ? UsdGenCross(a, UsdGenVec3d(1, 0, 0)) : UsdGenCross(a, UsdGenVec3d(0, 1, 0));
    }
    axis.Normalize();
    UsdGenVec3d const p(v);
    double const cs = std::cos(angle), sn = std::sin(angle);
    UsdGenVec3d const r = p * cs + UsdGenCross(axis, p) * sn + axis * (UsdGenDot(axis, p) * (1.0 - cs));
    return UsdGenVec3f(r);
}

/// Rest area of the faces a node scatters on: a GeomSubset's faces of its
/// parent mesh, or the whole mesh. 0 when there is no bound surface.
inline double RestSurfaceArea(UsdGenGraphSpec const *desc, UsdGenSurfaceId surface, bool hasSurface)
{
    UsdGenSpan<const UsdGenInputRecord> const surfaces =
        desc ? UsdGenSurfaceInputs(*desc) : UsdGenSpan<const UsdGenInputRecord>();
    if (!desc || !hasSurface || surface >= surfaces.size()) return 0.0;
    UsdGenSurfaceInputView const target = UsdGenSurfaceInputView::Of(surfaces[surface]);
    UsdGenSpan<const int> const subset = target.subsetFaces;
    bool const restricted = UsdGenSurfaceRestricted(target);
    UsdGenSurfaceInputView parentView;
    UsdGenSurfaceInputView const *mesh = &target;
    if (mesh->faceVertexCounts.empty()) {
        UsdGenNodeKey const parent = mesh->owner;
        mesh = nullptr;
        for (UsdGenInputRecord const &candidate : surfaces)
            if (parent.IsValid() && candidate.key == parent) {
                parentView = UsdGenSurfaceInputView::Of(candidate);
                mesh = &parentView;
                break;
            }
        if (!mesh) return 0.0;
    }
    UsdGenSpan<const UsdGenVec3f> const points = mesh->restPoints.empty() ? mesh->points : mesh->restPoints;
    std::vector<size_t> offsets(mesh->faceVertexCounts.size() + 1, 0);
    for (size_t f = 0; f < mesh->faceVertexCounts.size(); ++f)
        offsets[f + 1] = offsets[f] + size_t(std::max(0, mesh->faceVertexCounts[f]));
    if (offsets.back() != mesh->faceVertexIndices.size()) return 0.0;
    auto faceArea = [&](size_t f) {
        size_t const first = offsets[f], n = offsets[f + 1] - first;
        if (n < 3) return 0.0;
        UsdGenVec3d sum(0.0);
        for (size_t k = 0; k < n; ++k) {
            int const ia = mesh->faceVertexIndices[first + k];
            int const ib = mesh->faceVertexIndices[first + (k + 1) % n];
            if (ia < 0 || ib < 0 || size_t(ia) >= points.size() || size_t(ib) >= points.size())
                return 0.0;
            sum += UsdGenCross(UsdGenVec3d(points[ia]), UsdGenVec3d(points[ib]));
        }
        return 0.5 * sum.GetLength();
    };
    double area = 0.0;
    if (restricted) {
        for (int f : subset)
            if (f >= 0 && size_t(f) < mesh->faceVertexCounts.size()) area += faceArea(size_t(f));
    } else {
        for (size_t f = 0; f < mesh->faceVertexCounts.size(); ++f) area += faceArea(f);
    }
    return area;
}

/// The key a connected region/clump map value groups by: distinct 10-bit
/// levels (an 8-bit painted map) and distinct integer ids both separate.
inline int64_t RegionKey(double value)
{
    return std::isfinite(value) ? static_cast<int64_t>(std::llround(value * 1024.0))
                                : INT64_MIN;
}

}  // namespace opUtil
}  // namespace usdGen

#endif  // USDGEN_OP_UTIL_H
