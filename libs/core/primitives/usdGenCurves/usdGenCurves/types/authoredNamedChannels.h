#ifndef USDGEN_AUTHORED_NAMED_CHANNELS_H
#define USDGEN_AUTHORED_NAMED_CHANNELS_H

#include "curveBuffer.h"
#include "usdGenCurves/types/curveSetInput.h"
#include "usdGen/api/graphSpec.h"

#include <algorithm>
#include <cmath>
#include <climits>
#include <cstdint>
#include <iterator>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace usdGen {
namespace detail {
inline bool AuthoredNamedChannelFail(std::string const& message, std::string* reason) {
    if (reason) *reason = message;
    return false;
}

inline bool IsReservedAuthoredNamedChannel(UsdGenToken const& name) {
    static constexpr char const *names[] = {"points", "curveOffsets", "stableIds", "rest",
        "width", "widths", "restPoints", "hairT", "rootPrim", "rootUV",
        "sourceRootT", "sourceRootB", "sourceRootN", "sourceChunks"};
    std::string const &s = name.GetString();
    return std::any_of(std::begin(names), std::end(names),
                       [&](char const *n) { return s == n; });
}
inline bool AuthoredNamedChannelElements(UsdGenAuthoredPlaneView const& plane,
    size_t curves, size_t points, size_t* elements, std::string* reason) {
    switch (plane.domain) {
    case UsdGenAuthoredPlaneDomain::Point: *elements = points; return true;
    case UsdGenAuthoredPlaneDomain::Primitive: *elements = curves; return true;
    case UsdGenAuthoredPlaneDomain::Groom: *elements = 1; return true;
    default: return AuthoredNamedChannelFail("authored named channel has an invalid domain", reason);
    }
}
} // namespace detail

inline bool ValidateAuthoredNamedChannels(UsdGenCurveSetInputView const& source, size_t totalCurves,
    size_t totalPoints, std::string* reason = nullptr) {
    std::set<UsdGenToken> names;
    for (UsdGenAuthoredPlaneView const& plane : source.authoredPlanes) {
        if (plane.name.IsEmpty() || !names.insert(plane.name).second ||
            detail::IsReservedAuthoredNamedChannel(plane.name))
            return detail::AuthoredNamedChannelFail("authored named channel has an empty, duplicate, or reserved name", reason);
        if (plane.arity == 0 || plane.arity > 16)
            return detail::AuthoredNamedChannelFail("authored named channel arity is outside [1,16]", reason);
        size_t elements = 0;
        if (!detail::AuthoredNamedChannelElements(plane, totalCurves, totalPoints, &elements, reason) ||
            elements > std::numeric_limits<size_t>::max() / plane.arity)
            return detail::AuthoredNamedChannelFail("authored named channel cardinality overflows", reason);
        size_t const values = elements * plane.arity;
        if (plane.type == UsdGenAuthoredPlaneType::Float32) {
            if (!plane.intValues.empty() || plane.floatValues.size() != values)
                return detail::AuthoredNamedChannelFail("authored float named channel cardinality is invalid", reason);
            for (float value : plane.floatValues) if (!std::isfinite(value))
                return detail::AuthoredNamedChannelFail("authored float named channel contains a non-finite value", reason);
        } else if (plane.type == UsdGenAuthoredPlaneType::Int32) {
            if (!plane.floatValues.empty() || plane.intValues.size() != values)
                return detail::AuthoredNamedChannelFail("authored int named channel cardinality is invalid", reason);
        } else return detail::AuthoredNamedChannelFail("authored named channel has an invalid type", reason);
    }
    return true;
}

inline bool GatherAuthoredNamedChannels(UsdGenCurveSetInputView const& source,
    std::vector<uint32_t> const& sourceOffsets, std::vector<size_t> const& canonicalCurveOrder,
    UsdGenCurveBuffer* out, std::string* reason = nullptr) {
    if (!out || sourceOffsets.size() != canonicalCurveOrder.size() + 1 ||
        sourceOffsets.empty() || sourceOffsets.front() != 0 || sourceOffsets.back() != out->totalCvs ||
        canonicalCurveOrder.size() != out->totalCurves || out->totalCvs > INT_MAX ||
        out->cvOffsets.size() != canonicalCurveOrder.size() + 1 || out->cvOffsets.empty() ||
        out->cvOffsets.front() != 0 || out->cvOffsets.back() != int(out->totalCvs) ||
        !ValidateAuthoredNamedChannels(source, out->totalCurves, out->totalCvs, reason))
        return detail::AuthoredNamedChannelFail("authored named channel gather has invalid topology", reason);
    try {
        std::set<size_t> seen;
        for (size_t curve = 0; curve != canonicalCurveOrder.size(); ++curve) {
            size_t const sourceCurve = canonicalCurveOrder[curve];
            if (sourceCurve >= canonicalCurveOrder.size() || !seen.insert(sourceCurve).second ||
                sourceOffsets[sourceCurve] > sourceOffsets[sourceCurve + 1] ||
                out->cvOffsets[curve] < 0 || out->cvOffsets[curve] > out->cvOffsets[curve + 1] ||
                uint64_t(sourceOffsets[sourceCurve + 1] - sourceOffsets[sourceCurve]) !=
                    uint64_t(out->cvOffsets[curve + 1] - out->cvOffsets[curve]))
                return detail::AuthoredNamedChannelFail("authored named channel permutation is invalid", reason);
        }
        std::vector<UsdGenPlane> cv, curve;
        cv.reserve(source.authoredPlanes.size()); curve.reserve(source.authoredPlanes.size());
        for (UsdGenAuthoredPlaneView const& authored : source.authoredPlanes) {
            UsdGenPlane plane;
            plane.name = authored.name; plane.arity = authored.arity;
            bool const point = authored.domain == UsdGenAuthoredPlaneDomain::Point;
            bool const groom = authored.domain == UsdGenAuthoredPlaneDomain::Groom;
            plane.interpolation = point ? UsdGenTokens().vertex : groom ? UsdGenTokens().constant : UsdGenTokens().uniform;
            plane.type = authored.type == UsdGenAuthoredPlaneType::Float32 ? UsdGenTokens().float_ : UsdGenTokens().int_;
            size_t const elements = point ? out->totalCvs : groom ? 1 : out->totalCurves;
            size_t const values = elements * plane.arity;
            if (plane.type == UsdGenTokens().float_) plane.f = UsdGenFloatArray(values, 0.0f);
            else plane.i = UsdGenIntArray(values, 0);
            if (groom) {
                if (plane.type == UsdGenTokens().float_)
                    plane.f.assign(authored.floatValues.begin(), authored.floatValues.end());
                else plane.i.assign(authored.intValues.begin(), authored.intValues.end());
            } else for (size_t destinationCurve = 0; destinationCurve != canonicalCurveOrder.size(); ++destinationCurve) {
                size_t const sourceCurve = canonicalCurveOrder[destinationCurve];
                if (sourceCurve >= sourceOffsets.size() - 1 || sourceOffsets[sourceCurve] > sourceOffsets[sourceCurve + 1])
                    return detail::AuthoredNamedChannelFail("authored named channel permutation is invalid", reason);
                if (point) {
                    size_t const sourceBegin = size_t(sourceOffsets[sourceCurve]);
                    size_t const count = size_t(sourceOffsets[sourceCurve + 1] - sourceOffsets[sourceCurve]);
                    size_t const destinationBegin = size_t(out->cvOffsets[destinationCurve]);
                    if (destinationBegin + count > out->totalCvs || count > std::numeric_limits<size_t>::max() / plane.arity)
                        return detail::AuthoredNamedChannelFail("authored point named channel permutation is invalid", reason);
                    if (plane.type == UsdGenTokens().float_)
                        std::copy_n(authored.floatValues.cdata() + sourceBegin * plane.arity, count * plane.arity,
                            plane.f.begin() + destinationBegin * plane.arity);
                    else std::copy_n(authored.intValues.cdata() + sourceBegin * plane.arity, count * plane.arity,
                        plane.i.begin() + destinationBegin * plane.arity);
                } else if (plane.type == UsdGenTokens().float_)
                    std::copy_n(authored.floatValues.cdata() + sourceCurve * plane.arity, plane.arity,
                        plane.f.begin() + destinationCurve * plane.arity);
                else std::copy_n(authored.intValues.cdata() + sourceCurve * plane.arity, plane.arity,
                    plane.i.begin() + destinationCurve * plane.arity);
            }
            (point ? cv : curve).push_back(std::move(plane));
        }
        auto byName=[](UsdGenPlane const& a, UsdGenPlane const& b) { return a.name < b.name; };
        std::sort(cv.begin(), cv.end(), byName); std::sort(curve.begin(), curve.end(), byName);
        UsdGenCurveBuffer candidate;
        candidate.totalCurves = out->totalCurves;
        candidate.totalCvs = out->totalCvs;
        candidate.extraCv = std::move(cv); candidate.extraCurve = std::move(curve);
        if (!detail::_UsdGenValidateExtraPlanes(candidate, reason)) return false;
        out->extraCv = std::move(candidate.extraCv); out->extraCurve = std::move(candidate.extraCurve);
        return true;
    } catch (...) { return detail::AuthoredNamedChannelFail("authored named channel gather allocation failed", reason); }
}
} // namespace usdGen
#endif
