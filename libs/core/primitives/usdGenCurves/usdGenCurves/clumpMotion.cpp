// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGen/clumpMotion.h"
#include "usdGen/registries/engineTokens.h"
#include "usdGenCurves/types/curveSet.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <tuple>


namespace usdGen {
namespace {

bool Fail(std::string const &message, std::string *error)
{
    if (error) *error = message;
    return false;
}

bool ParseLevel(UsdGenToken const &name, char const *prefix, int *level)
{
    std::string const &s = name.GetString();
    size_t const n = std::strlen(prefix);
    if (s.compare(0, n, prefix) != 0 || s.size() == n) return false;
    if (s[n] == '0' && s.size() > n + 1) return false;
    int value = 0;
    for (size_t i = n; i < s.size(); ++i) {
        if (s[i] < '0' || s[i] > '9' ||
            value > (std::numeric_limits<int>::max() - (s[i] - '0')) / 10)
            return false;
        value = value * 10 + (s[i] - '0');
    }
    *level = value;
    return true;
}

UsdGenPlane const *Find(std::vector<UsdGenPlane> const &planes,
                        std::string const &name)
{
    auto const it = std::find_if(planes.begin(), planes.end(),
        [&](UsdGenPlane const &p) { return p.name.GetString() == name; });
    return it == planes.end() ? nullptr : &*it;
}

uint64_t DecodeId(int32_t low, int32_t high)
{
    uint32_t lo, hi;
    std::memcpy(&lo, &low, sizeof(lo));
    std::memcpy(&hi, &high, sizeof(hi));
    return uint64_t(lo) | (uint64_t(hi) << 32);
}

} // namespace

bool UsdGenBuildClumpMotion(UsdGenCurveBuffer const &buffer,
                            UsdGenClumpMotion *out, std::string *error,
                            UsdGenClumpMotion const *previous)
{
    if (!out) return Fail("clump motion output is null", error);
    UsdGenClumpMotion candidate;
    for (UsdGenPlane const &plane : buffer.extraCurve) {
        int level = 0;
        std::string const &name = plane.name.GetString();
        if (name.rfind("clumpWeight_", 0) == 0)
            return Fail("clumpWeight plane is in the uniform domain: " + name, error);
        for (char const *prefix : {"clumpId_", "clumpCenter_", "clumpCenterId_"})
            if (name.rfind(prefix, 0) == 0 && !ParseLevel(plane.name, prefix, &level))
                return Fail("invalid clump motion plane name " + name, error);
        if ((ParseLevel(plane.name, "clumpCenter_", &level) ||
             ParseLevel(plane.name, "clumpCenterId_", &level)) &&
            !Find(buffer.extraCurve, "clumpId_" + std::to_string(level)))
            return Fail("orphan clump motion plane " + plane.name.GetString(), error);
    }
    for (UsdGenPlane const &plane : buffer.extraCv) {
        int level = 0;
        std::string const &name = plane.name.GetString();
        if (name.rfind("clumpId_", 0) == 0 ||
            name.rfind("clumpCenter_", 0) == 0 ||
            name.rfind("clumpCenterId_", 0) == 0)
            return Fail("uniform clump plane is in the vertex domain: " + name, error);
        if (plane.name.GetString().rfind("clumpWeight_", 0) == 0 &&
            !ParseLevel(plane.name, "clumpWeight_", &level))
            return Fail("invalid clump motion plane name " + plane.name.GetString(), error);
        if (ParseLevel(plane.name, "clumpWeight_", &level) &&
            !Find(buffer.extraCurve, "clumpId_" + std::to_string(level)))
            return Fail("orphan clump motion plane " + plane.name.GetString(), error);
    }
    UsdGenPoolVector<int> seenLevels;   // empty unless clump planes exist; no node alloc
    for (UsdGenPlane const &ids : buffer.extraCurve) {
        int level = 0;
        if (!ParseLevel(ids.name, "clumpId_", &level)) continue;
        if (std::find(seenLevels.begin(), seenLevels.end(), level) != seenLevels.end())
            return Fail("duplicate clump level " + std::to_string(level), error);
        seenLevels.push_back(level);
        if (ids.interpolation != UsdGenTokens().uniform ||
            ids.type != UsdGenTokens().int_ || ids.arity != 1 ||
            ids.i.size() != buffer.totalCurves)
            return Fail("invalid clumpId plane " + ids.name.GetString(), error);

        UsdGenClumpMotionLevel item;
        item.level = level;
        std::string const suffix = std::to_string(level);
        UsdGenPlane const *anchor = Find(buffer.extraCurve, "clumpCenter_" + suffix);
        UsdGenPlane const *centerId = Find(buffer.extraCurve, "clumpCenterId_" + suffix);
        UsdGenPlane const *weight = Find(buffer.extraCv, "clumpWeight_" + suffix);
        int const present = int(anchor != nullptr) + int(centerId != nullptr) + int(weight != nullptr);
        if (present != 0 && present != 3)
            return Fail("incomplete clump motion planes for level " + suffix, error);
        if (present == 3) {
            if (anchor->interpolation != UsdGenTokens().uniform ||
                anchor->type != UsdGenTokens().float_ || anchor->arity != 3 ||
                anchor->f.size() != size_t(buffer.totalCurves) * 3 ||
                centerId->interpolation != UsdGenTokens().uniform ||
                centerId->type != UsdGenTokens().int_ || centerId->arity != 2 ||
                centerId->i.size() != size_t(buffer.totalCurves) * 2 ||
                weight->interpolation != UsdGenTokens().vertex ||
                weight->type != UsdGenTokens().float_ || weight->arity != 1 ||
                weight->f.size() != buffer.totalCvs)
                return Fail("invalid clump motion layout for level " + suffix, error);
            for (float w : weight->f)
                if (!std::isfinite(w) || w < 0.0f || w > 1.0f)
                    return Fail("invalid clump motion weight for level " + suffix, error);
            item.weight = weight->f; // immutable COW capture copy
            item.sourceMembership = ids.i;
            item.sourceAnchor = anchor->f;
            item.sourceCenterId = centerId->i;
            // The prior capture owns the CoW arrays, so a matching allocation
            // cannot have been freed and reused at the same address. Layout
            // was checked above; new content detaches on mutation and misses.
            UsdGenClumpMotionLevel const *cached = nullptr;
            if (previous)
                for (auto const &prior : previous->levels)
                    if (prior.level == level &&
                        prior.groupForCurve.size() == buffer.totalCurves &&
                        prior.sourceMembership.size() == ids.i.size() &&
                        prior.sourceMembership.cdata() == ids.i.cdata() &&
                        prior.sourceAnchor.size() == anchor->f.size() &&
                        prior.sourceAnchor.cdata() == anchor->f.cdata() &&
                        prior.sourceCenterId.size() == centerId->i.size() &&
                        prior.sourceCenterId.cdata() == centerId->i.cdata()) {
                        cached = &prior;
                        break;
                    }
            if (cached) {
                item.groupForCurve = cached->groupForCurve;
                item.groups = cached->groups;
                candidate.levels.push_back(std::move(item));
                continue;
            }
            std::vector<uint32_t> groupForCurve(buffer.totalCurves, UINT32_MAX);
            std::vector<UsdGenClumpMotionGroup> groups;
            // Sorting by stable center identity builds dense group indices once.
            std::vector<std::tuple<uint64_t, int32_t, uint32_t>> members;
            members.reserve(buffer.totalCurves);
            std::map<int32_t, uint64_t> centerForMembership;
            for (uint32_t c = 0; c < buffer.totalCurves; ++c) {
                if (ids.i[c] < 0) continue;
                uint64_t const key = DecodeId(centerId->i[size_t(c) * 2],
                                              centerId->i[size_t(c) * 2 + 1]);
                if (auto [it, inserted] = centerForMembership.emplace(ids.i[c], key);
                    !inserted && it->second != key)
                    return Fail("one clump membership has multiple centers at level " +
                                suffix, error);
                members.emplace_back(key, ids.i[c], c);
            }
            std::sort(members.begin(), members.end());
            uint64_t previousKey = 0;
            int32_t previousMembership = -1;
            for (auto const &[key, membership, c] : members) {
                UsdGenVec3f const rest(anchor->f[size_t(c) * 3],
                                   anchor->f[size_t(c) * 3 + 1],
                                   anchor->f[size_t(c) * 3 + 2]);
                if (!std::isfinite(rest[0]) || !std::isfinite(rest[1]) ||
                    !std::isfinite(rest[2]))
                    return Fail("nonfinite clump center for level " + suffix, error);
                if (groups.empty() || key != previousKey) {
                    groups.push_back({rest, key});
                    previousKey = key;
                    previousMembership = membership;
                } else if (previousMembership != membership ||
                           groups.back().restAnchor != rest) {
                    return Fail("inconsistent clump center for level " + suffix, error);
                }
                groupForCurve[c] = uint32_t(groups.size() - 1);
            }
            item.groupForCurve = UsdGenClumpReadOnlyArray<uint32_t>(
                std::move(groupForCurve));
            item.groups = UsdGenClumpReadOnlyArray<UsdGenClumpMotionGroup>(
                std::move(groups));
        } else {
            item.groupForCurve = UsdGenClumpReadOnlyArray<uint32_t>(
                std::vector<uint32_t>(buffer.totalCurves, UINT32_MAX));
        }
        candidate.levels.push_back(std::move(item));
    }
    std::sort(candidate.levels.begin(), candidate.levels.end(),
        [](auto const &a, auto const &b) { return a.level < b.level; });
    *out = std::move(candidate);
    return true;
}

bool UsdGenBuildClumpMotion(UsdGenCurveSetView const &curves,
                            UsdGenClumpMotion *out,
                            std::string *error,
                            UsdGenClumpMotion const *previous)
{
    return UsdGenBuildClumpMotion(UsdGenCurvePlanesBuffer(curves), out, error, previous);
}

} // namespace usdGen
