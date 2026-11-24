// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT

#ifndef USDGEN_VULKAN_SOURCE_VALIDATION_H
#define USDGEN_VULKAN_SOURCE_VALIDATION_H

#include "sourceGeneration.h"
#include "usdGen/registries/engineTokens.h"
#include <cmath>
#include <cstring>
#include <limits>
#include <set>

namespace usdGen::vulkan {
using SourceHostPlane = VulkanPreparedSource::Plane;

// Host capture validation only: no native allocation, submission or readback.
inline bool PackSource(VulkanSourcePrepareInfo const& in,
    std::vector<SourceHostPlane>* planes, UsdGenDeviceGeometryMetadata* geometry,
    std::string* reason) {
    auto fail = [&](char const* text) { if (reason) *reason = text; return false; };
    if (!planes || !geometry) return fail("missing source capture output");
    auto const& b = in.source;
    size_t const c = b.totalCurves, p = b.totalCvs;
    if ((!c && p) || b.px.size() != p || b.py.size() != p || b.pz.size() != p ||
        b.curveId.size() != c) return fail("source required channel cardinality mismatch");
    auto optionalSize = [](size_t actual, size_t expected) { return !actual || actual == expected; };
    if (!optionalSize(b.rest.size(), p) || !optionalSize(b.width.size(), p) ||
        !optionalSize(b.hairT.size(), p) || !optionalSize(b.rootPrim.size(), c) ||
        !optionalSize(b.rootUV.size(), c))
        return fail("source optional channel cardinality mismatch");
    bool frames = !b.rootT.empty() || !b.rootB.empty() || !b.rootN.empty();
    if (frames && (b.rootT.size() != c || b.rootB.size() != c || b.rootN.size() != c))
        return fail("source frame trio is incomplete");
    std::set<uint64_t> stableIds;
    for (size_t i = 0; i != c; ++i)
        if (!stableIds.insert(b.curveId[i]).second) return fail("source stable IDs must be unique");
    if (!detail::_UsdGenValidateExtraPlanes(b, reason)) return false;
    auto finite = [](float v) { return std::isfinite(v); };
    for (size_t i = 0; i != p; ++i) {
        if (!finite(b.px[i]) || !finite(b.py[i]) || !finite(b.pz[i])) return fail("nonfinite source point");
        if (!b.width.empty() && (!finite(b.width[i]) || b.width[i] < 0)) return fail("invalid source width");
        if (!b.hairT.empty() && !finite(b.hairT[i])) return fail("nonfinite source hairT");
        if (!b.rest.empty()) for (int a = 0; a != 3; ++a)
            if (!finite(b.rest[i][a])) return fail("nonfinite source rest point");
    }
    for (size_t i = 0; i != c; ++i) {
        if (!b.rootUV.empty() && (!finite(b.rootUV[i][0]) || !finite(b.rootUV[i][1]))) return fail("nonfinite root UV");

        if (frames) for (int a = 0; a != 3; ++a)
            if (!finite(b.rootT[i][a]) || !finite(b.rootB[i][a]) || !finite(b.rootN[i][a]))
                return fail("nonfinite source frame");
    }
    std::vector<uint32_t> offsets(c + 1, 0);
    if (b.cvOffsets.empty()) {
        if (c && p % c) return fail("implicit source topology is nonuniform");
        for (size_t i = 0; i <= c; ++i) offsets[i] = uint32_t(c ? i * (p / c) : 0);
    } else {
        if (b.cvOffsets.size() != c + 1 || b.cvOffsets[0] != 0)
            return fail("invalid source offset domain");
        for (size_t i = 0; i <= c; ++i) {
            if (b.cvOffsets[i] < 0 || (i && b.cvOffsets[i] < b.cvOffsets[i-1]))
                return fail("invalid source offset order");
            offsets[i] = uint32_t(b.cvOffsets[i]);
        }
        if (offsets.back() != p) return fail("source offsets do not span points");
    }
    for (auto const& chunk : b.chunks) {
        if (chunk.firstCurve > c || chunk.curveCount > c - chunk.firstCurve ||
            chunk.liveCount > chunk.curveCount || chunk.firstCv > p ||
            chunk.firstCv != offsets[chunk.firstCurve]) return fail("invalid source chunk range");
        if (chunk.cvCount) for (size_t i = chunk.firstCurve; i < size_t(chunk.firstCurve) + chunk.curveCount; ++i)
            if (offsets[i+1] - offsets[i] != chunk.cvCount) return fail("chunk uniform span disagrees with offsets");
    }
    auto g = in.geometry;
    if ((g.curveCount && g.curveCount != c) || (g.pointCount && g.pointCount != p) ||
        (g.topologyVersion && g.topologyVersion != b.topologyVersion) ||
        (g.valueVersion && g.valueVersion != b.valueVersion)) return fail("source capture metadata mismatch");
    g.curveCount = c; g.pointCount = p; g.topologyVersion = b.topologyVersion; g.valueVersion = b.valueVersion;
    std::vector<SourceHostPlane> packed;
    using T = UsdGenDeviceValueType; using D = UsdGenDeviceDomain; using S = UsdGenDeviceChannelSemantic;
    auto add = [&](char const* name, T type, D domain, size_t elements, uint32_t arity,
                   S semantic, void const* bytes, size_t count, bool privateFrame = false) {
        SourceHostPlane item;
        item.metadata = {name, type, domain, elements, arity, uint32_t((type == T::UInt64 ? 8 : 4) * arity), true, semantic};
        item.privateFrame = privateFrame;
        item.bytes.resize(count);
        if (count) std::memcpy(item.bytes.data(), bytes, count);
        packed.push_back(std::move(item));
    };
    std::vector<float> points(p * 3);
    for (size_t i = 0; i != p; ++i) { points[3*i] = b.px[i]; points[3*i+1] = b.py[i]; points[3*i+2] = b.pz[i]; }
    add("points", T::Float32x3, D::Point, p, 3, S::Points, points.data(), points.size()*4);
    add("curveOffsets", T::UInt32, D::Topology, c+1, 1, S::CurveOffsets, offsets.data(), offsets.size()*4);
    add("stableIds", T::UInt64, D::Primitive, c, 1, S::StableIds, b.curveId.cdata(), c*8);
    auto optional = [&](char const* name, T type, D domain, size_t n, uint32_t arity, S semantic, auto const& array, bool privateFrame = false) {
        if (!array.empty()) add(name, type, domain, n, arity, semantic, array.cdata(), array.size()*sizeof(array[0]), privateFrame);
    };
    optional("rest",T::Float32x3,D::Point,p,3,S::RestPoints,b.rest);
    optional("width",T::Float32,D::Point,p,1,S::Widths,b.width);
    optional("hairT",T::Float32,D::Point,p,1,S::HairT,b.hairT);
    if (!p) {
        // Empty CUDA generations retain the complete core channel schema.
        add("rest", T::Float32x3, D::Point, 0, 3, S::RestPoints, nullptr, 0);
        add("width", T::Float32, D::Point, 0, 1, S::Widths, nullptr, 0);
        add("hairT", T::Float32, D::Point, 0, 1, S::HairT, nullptr, 0);
    }
    optional("rootPrim",T::Int32,D::Primitive,c,1,S::RootPrim,b.rootPrim);
    optional("rootUV",T::Float32x2,D::Primitive,c,2,S::RootUV,b.rootUV);

    optional("sourceRootT",T::Float32x3,D::Primitive,c,3,S::Generic,b.rootT,true);
    optional("sourceRootB",T::Float32x3,D::Primitive,c,3,S::Generic,b.rootB,true);
    optional("sourceRootN",T::Float32x3,D::Primitive,c,3,S::Generic,b.rootN,true);
    std::set<std::string> names{"points","curveOffsets","stableIds","rest","width","widths","restPoints","hairT","rootPrim","rootUV","sourceRootT","sourceRootB","sourceRootN","sourceChunks"};
    auto named = [&](VulkanSourceNamedChannel const& channel) {
        if (channel.metadata.semantic != S::Generic || !names.insert(channel.metadata.name).second)
            return fail("source named channel collides with a reserved or authored field");
        if (channel.privateStructural &&
            (channel.metadata.domain != D::Primitive ||
             !((channel.metadata.type == T::Float32x3 && channel.metadata.arity == 3) ||
               (channel.metadata.type == T::Float32 && channel.metadata.arity == 1)) ||
             channel.metadata.elementCount != c))
            return fail("invalid private source primitive field");
        packed.push_back({channel.metadata, channel.bytes, channel.privateStructural}); return true;
    };
    auto extra = [&](UsdGenPlane const& plane, bool cv) {
        UsdGenEngineTokens const& t = UsdGenTokens();
        if ((plane.type != t.float_ && plane.type != t.int_) ||
            (plane.interpolation != t.vertex && plane.interpolation != t.uniform && plane.interpolation != t.constant) ||
            (cv && plane.interpolation != t.vertex) || (!cv && plane.interpolation == t.vertex) ||
            (plane.type == t.float_ ? !plane.i.empty() : !plane.f.empty())) return fail("invalid CPU named plane schema");
        D domain = plane.interpolation == t.vertex ? D::Point : plane.interpolation == t.uniform ? D::Primitive : D::Groom;
        VulkanSourceNamedChannel channel;
        channel.metadata = {plane.name.GetString(), plane.type == t.float_ ? T::Float32 : T::Int32,
            domain, domain == D::Point ? p : domain == D::Primitive ? c : 1, plane.arity, uint32_t(4*plane.arity), true, S::Generic};
        size_t count = plane.type == t.float_ ? plane.f.size()*4 : plane.i.size()*4;
        channel.bytes.resize(count);
        if (count) std::memcpy(channel.bytes.data(), plane.type == t.float_ ? static_cast<void const*>(plane.f.cdata()) : static_cast<void const*>(plane.i.cdata()), count);
        return named(channel);
    };
    for (auto const& plane : b.extraCv) if (!extra(plane, true)) return false;
    for (auto const& plane : b.extraCurve) if (!extra(plane, false)) return false;
    for (auto const& channel : in.additionalNamed) if (!named(channel)) return false;
    std::vector<UsdGenDeviceChannelMetadata> metadata;
    for (auto const& plane : packed) metadata.push_back(plane.metadata);
    if (!ValidateUsdGenDeviceMetadata(g, metadata, reason)) return false;
    for (auto const& plane : packed) {
        size_t span = 0;
        if (!UsdGenDeviceChannelStorageBytes(plane.metadata, &span, reason) || span != plane.bytes.size())
            return fail("source typed byte payload does not match its strided metadata");
    }
    *planes = std::move(packed); *geometry = std::move(g); return true;
}
} // namespace usdGen::vulkan
#endif
