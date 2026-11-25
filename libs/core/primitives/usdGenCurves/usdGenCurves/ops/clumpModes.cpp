// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — reference clump modes. See clumpModes.h.
#include "usdGen/ops/clumpModes.h"

#include "usdGen/ops/opUtil.h"

#include <nanoflann.hpp>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <map>
#include <numeric>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace clumpRef {

namespace {

constexpr float kTwoPi = 6.28318530717958647692f;
// Candidate lists up to this size are scanned; larger ones use a kd-tree.
constexpr size_t kBruteForceLimit = 64;
constexpr size_t kKnn = 16;

inline float const *At(float const *xyz, uint32_t i) { return xyz + 3 * size_t(i); }

using strand::Clamp01;
using strand::Load;
using strand::Store;

}  // namespace

// ---------------------------------------------------------------------------
// Profile and flyaways
// ---------------------------------------------------------------------------

float ClumpWeight(float u, float length, float amount, BezierControls const &k)
{
    float const r = Clamp01(k.root), m = Clamp01(k.mid), t = Clamp01(k.tip);
    float y;
    if (length <= 1e-6f) {
        y = m;
    } else {
        // The reference passes s = u * L and L and divides again.
        float const x = Clamp01((u * length) / length);
        y = strand::BezierProfile(x, r, m, k.midShift, t);
    }
    return Clamp01(y * Clamp01(amount));
}

bool IsFlyaway(uint64_t strandIndex, float fraction)
{
    double const p = double(Clamp01(fraction));
    if (!(p > 0.0)) return false;
    uint64_t const z = strand::SplitMix64(strandIndex);
    return p > double(z >> 11) * (1.0 / 9007199254740992.0);
}

// ---------------------------------------------------------------------------
// Nearest-point queries
// ---------------------------------------------------------------------------

namespace {

struct Cloud
{
    float const *data = nullptr;
    size_t count = 0;
    size_t kdtree_get_point_count() const { return count; }
    float kdtree_get_pt(size_t i, size_t d) const { return data[i * 3 + d]; }
    template <class B> bool kdtree_get_bbox(B &) const { return false; }
};
using CloudIndex = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<float, Cloud>, Cloud, 3, uint32_t>;

}  // namespace

struct NearestFinder::Tree
{
    Cloud cloud;
    CloudIndex index;
    explicit Tree(Cloud c)
        : cloud(c), index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10)) {}
};

NearestFinder::NearestFinder() = default;
NearestFinder::~NearestFinder() = default;
NearestFinder::NearestFinder(NearestFinder &&) noexcept = default;
NearestFinder &NearestFinder::operator=(NearestFinder &&) noexcept = default;

void NearestFinder::Build(std::vector<float> points)
{
    _tree.reset();
    _points = std::move(points);
    size_t const n = _points.size() / 3;
    // The tree points at the vector's heap buffer, which a move keeps.
    if (n > kBruteForceLimit) _tree = std::make_unique<Tree>(Cloud{_points.data(), n});
}

int32_t NearestFinder::Nearest(float const *q, float *d2) const
{
    size_t const n = Size();
    int32_t best = -1;
    float bestD = 0.0f;
    auto consider = [&](size_t i) {
        float const d = Dist2(q, &_points[3 * i]);
        if (best < 0 || d < bestD || (d == bestD && int32_t(i) < best)) {
            best = int32_t(i);
            bestD = d;
        }
    };
    if (!_tree) {
        for (size_t i = 0; i < n; ++i) consider(i);
    } else {
        size_t const k = std::min(kKnn, n);
        uint32_t found[kKnn];
        float distances[kKnn];
        size_t const got = _tree->index.knnSearch(q, k, found, distances);
        // Exact refinement: the same float metric, ties to the lower index.
        for (size_t j = 0; j < got; ++j) consider(found[j]);
    }
    if (d2) *d2 = bestD;
    return best;
}

// ---------------------------------------------------------------------------
// Seeds
// ---------------------------------------------------------------------------

std::vector<uint32_t> RandomSeedCenters(std::vector<uint32_t> const &A, float const *roots,
                                        int const *partKey, size_t count, std::mt19937 &rng)
{
    size_t const N = A.size();
    size_t const K = std::min(count, N);
    std::vector<uint32_t> centers;
    if (K == 0) return centers;
    centers.reserve(K);
    bool const many = K > 255;
    // Farthest-point distances by position in A, shared by every group.
    std::vector<float> minD2(many ? 0 : N, FLT_MAX);
    auto add = [&](size_t pos) {
        uint32_t const c = A[pos];
        centers.push_back(c);
        if (many) return;
        float const *pc = At(roots, c);
        for (size_t x = 0; x < N; ++x) {
            float const d = Dist2(At(roots, A[x]), pc);
            if (d < minD2[x]) minD2[x] = d;
        }
    };
    auto farthest = [&](std::vector<size_t> const &positions, size_t quota) {
        add(positions[size_t(rng() % positions.size())]);
        for (size_t picked = 1; picked < quota; ++picked) {
            float best = -1.0f;
            size_t bi = positions.front();
            for (size_t pos : positions) {
                if (minD2[pos] > best) {
                    best = minD2[pos];
                    bi = pos;
                }
            }
            add(bi);
        }
    };
    auto shuffled = [&](std::vector<size_t> positions, size_t quota) {
        strand::MsvcShuffle(&positions, rng);
        for (size_t j = 0; j < quota && j < positions.size(); ++j) add(positions[j]);
    };

    if (partKey) {
        std::map<int, std::vector<size_t>> groups;
        for (size_t pos = 0; pos < N; ++pos) groups[partKey[A[pos]]].push_back(pos);
        if (groups.size() > 1) {
            size_t j = 0;
            for (auto const &entry : groups) {
                ++j;
                size_t const remaining = K - centers.size();
                size_t q = j == groups.size()
                    ? remaining
                    : std::max<size_t>(1, size_t(std::llround(
                          double(entry.second.size()) * double(K) / double(N))));
                q = std::min({q, remaining, entry.second.size()});
                if (q == 0) continue;
                if (many) shuffled(entry.second, q);
                else farthest(entry.second, q);
            }
        }
    }
    if (centers.empty()) {
        std::vector<size_t> all(N);
        std::iota(all.begin(), all.end(), size_t(0));
        if (many) shuffled(std::move(all), K);
        else farthest(all, K);
    }
    return centers;
}

void SampleBucket(std::vector<uint32_t> const &bucket, size_t quota, float const *roots,
                  std::mt19937 &rng, std::vector<char> *chosen,
                  std::vector<uint32_t> *centers)
{
    size_t const m = bucket.size();
    if (m == 0 || quota == 0) return;
    if (quota >= 256) {
        std::vector<uint32_t> order = bucket;
        strand::MsvcShuffle(&order, rng);
        size_t taken = 0;
        for (uint32_t c : order) {
            if (taken >= quota) break;
            if ((*chosen)[c]) continue;
            (*chosen)[c] = 1;
            centers->push_back(c);
            ++taken;
        }
        return;
    }
    std::vector<float> minD2(m, FLT_MAX);
    auto add = [&](size_t pos) {
        uint32_t const c = bucket[pos];
        (*chosen)[c] = 1;
        centers->push_back(c);
        float const *pc = At(roots, c);
        for (size_t x = 0; x < m; ++x) {
            float const d = Dist2(At(roots, bucket[x]), pc);
            if (d < minD2[x]) minD2[x] = d;
        }
    };
    add(size_t(rng() % m));
    for (size_t picked = 1; picked < quota; ++picked) {
        float best = -1.0f;
        size_t bi = m;
        for (size_t x = 0; x < m; ++x) {
            if ((*chosen)[bucket[x]]) continue;
            if (minD2[x] > best) {
                best = minD2[x];
                bi = x;
            }
        }
        if (bi == m) break;
        add(bi);
    }
}

std::vector<uint32_t> GuideSeededCenters(std::vector<uint32_t> const &A, float const *roots,
                                         int const *nearestGuide, float const *guideRoots,
                                         size_t guideCount, int const *partKey,
                                         size_t count, std::mt19937 &rng)
{
    size_t const N = A.size();
    size_t const K = std::min(count, N);
    std::vector<uint32_t> centers;
    if (K == 0) return centers;
    uint32_t curves = 0;
    for (uint32_t c : A) curves = std::max(curves, c + 1);
    std::vector<char> chosen(curves, 0);

    if (guideCount > 0 && nearestGuide && guideRoots) {
        size_t const G = guideCount;
        std::vector<std::vector<uint32_t>> buckets(G);
        for (uint32_t c : A) {
            int const g = nearestGuide[c];
            if (g >= 0 && size_t(g) < G) buckets[size_t(g)].push_back(c);
        }
        std::vector<size_t> nonEmpty;
        for (size_t g = 0; g < G; ++g)
            if (!buckets[g].empty()) nonEmpty.push_back(g);
        std::vector<size_t> quota(G, 0);
        if (!nonEmpty.empty() && nonEmpty.size() < K) {
            for (size_t g : nonEmpty) quota[g] = 1;
            size_t remaining = K - nonEmpty.size();
            while (remaining > 0) {
                double best = -1.0;
                size_t bg = G;
                for (size_t g : nonEmpty) {
                    if (quota[g] >= buckets[g].size()) continue;
                    double const score = double(buckets[g].size()) / double(quota[g] + 1);
                    if (score > best) {
                        best = score;
                        bg = g;
                    }
                }
                if (bg == G) break;
                ++quota[bg];
                --remaining;
            }
        } else if (!nonEmpty.empty()) {
            size_t first = nonEmpty.front();
            for (size_t g : nonEmpty)
                if (buckets[g].size() > buckets[first].size()) first = g;
            quota[first] = 1;
            size_t picked = 1;
            std::vector<float> minDist(G, FLT_MAX);
            size_t last = first;
            while (picked < K) {
                for (size_t g : nonEmpty) {
                    float const d = std::sqrt(Dist2(At(guideRoots, uint32_t(g)),
                                                    At(guideRoots, uint32_t(last))));
                    if (d < minDist[g]) minDist[g] = d;
                }
                double best = -1.0;
                size_t bg = G;
                for (size_t g : nonEmpty) {
                    if (quota[g] > 0) continue;
                    double const score =
                        std::sqrt(double(buckets[g].size())) * double(minDist[g]);
                    if (score > best) {
                        best = score;
                        bg = g;
                    }
                }
                if (bg == G) break;
                quota[bg] = 1;
                last = bg;
                ++picked;
            }
        }
        for (size_t g = 0; g < G; ++g)
            if (quota[g] > 0) SampleBucket(buckets[g], quota[g], roots, rng, &chosen, &centers);
    } else if (partKey) {
        std::map<int, std::vector<uint32_t>> groups;
        for (uint32_t c : A) groups[partKey[c]].push_back(c);
        if (groups.size() > 1) {
            size_t j = 0;
            for (auto const &entry : groups) {
                ++j;
                size_t const remaining = K - centers.size();
                size_t q = j == groups.size()
                    ? remaining
                    : std::max<size_t>(1, size_t(std::llround(
                          double(entry.second.size()) * double(K) / double(N))));
                q = std::min(q, remaining);
                if (q == 0) continue;
                SampleBucket(entry.second, q, roots, rng, &chosen, &centers);
            }
        }
    }
    if (centers.empty()) SampleBucket(A, K, roots, rng, &chosen, &centers);
    return centers;
}

// ---------------------------------------------------------------------------
// Assignment
// ---------------------------------------------------------------------------

void AssignCenters(std::vector<uint32_t> const &A, float const *roots,
                   std::vector<uint32_t> const &centers,
                   std::vector<std::vector<uint32_t>> const &groups,
                   std::vector<uint32_t> const &groupOf,
                   std::vector<int32_t> *clumpOf)
{
    size_t const N = A.size();
    std::vector<NearestFinder> finders(groups.size());
    for (size_t g = 0; g < groups.size(); ++g) {
        std::vector<float> points;
        points.reserve(groups[g].size() * 3);
        for (uint32_t k : groups[g]) {
            float const *p = At(roots, centers[k]);
            points.insert(points.end(), p, p + 3);
        }
        finders[g].Build(std::move(points));
    }
    for (size_t pos = 0; pos < N; ++pos) {
        uint32_t const g = groupOf[pos];
        if (g >= groups.size()) continue;
        int32_t const k = finders[g].Nearest(At(roots, A[pos]));
        (*clumpOf)[A[pos]] = k < 0 ? -1 : int32_t(groups[g][size_t(k)]);
    }
}

// ---------------------------------------------------------------------------
// Guide curves
// ---------------------------------------------------------------------------

bool GuideTable::Build(UsdGenCurveBuffer const &guides, size_t ng, std::string *error)
{
    std::vector<uint32_t> spans;
    if (!opUtil::CurveSpans(guides, &spans, error)) return false;
    size_t const total = guides.totalCvs;
    if (guides.px.size() != total || guides.py.size() != total || guides.pz.size() != total) {
        if (error) *error = "guide point planes do not match the CV count";
        return false;
    }
    _ng = std::max<size_t>(ng, 2);
    size_t const G = guides.totalCurves;
    _table.assign(3 * _ng * G, 0.0f);
    _length.assign(G, 0.0f);
    std::vector<float> points, cum;
    for (size_t g = 0; g < G; ++g) {
        size_t const first = spans[g], n = spans[g + 1] - spans[g];
        if (n == 0) continue;
        points.resize(3 * n);
        for (size_t i = 0; i < n; ++i) {
            points[3 * i] = guides.px[first + i];
            points[3 * i + 1] = guides.py[first + i];
            points[3 * i + 2] = guides.pz[first + i];
        }
        float const L = strand::ArcLengths(points.data(), n, &cum);
        _length[g] = L;
        strand::ResampleArc(points.data(), n, cum, 0.0f, L, _ng, &_table[3 * _ng * g]);
    }
    return true;
}

GfVec3f GuideTable::Sample(size_t g, float a) const
{
    float const *t = &_table[3 * _ng * g];
    float const L = _length[g];
    if (!(L > 1e-6f)) return Load(t, 0);
    if (a > L) {
        GfVec3f const last = Load(t, _ng - 1);
        GfVec3f const dir = strand::Normalize(last - Load(t, _ng - 2), GfVec3f(0.0f));
        return last + dir * (a - L);
    }
    float const idx = Clamp01(a / L) * float(_ng - 1);
    size_t const i0 = std::min(size_t(idx), _ng - 2);
    float const f = idx - float(i0);
    GfVec3f const p0 = Load(t, i0), p1 = Load(t, i0 + 1);
    return p0 + (p1 - p0) * f;
}

void SynthesizeCenterCurve(GuideTable const &table, size_t g, GfVec3f const &root,
                           float length, size_t n, float *out)
{
    if (n == 0) return;
    Store(out, 0, root);
    GfVec3f dir(0.0f);
    bool haveDir = false;
    GfVec3f previous = table.Sample(g, 0.0f);
    for (size_t i = 1; i < n; ++i) {
        float const a = float(i) / float(n - 1) * length;
        GfVec3f const p = table.Sample(g, a);
        GfVec3f seg = p - previous;
        previous = p;
        float const segLength = seg.GetLength();
        if (haveDir && GfDot(seg, dir) < 0.0f) seg = seg * 0.25f + dir * (0.75f * segLength);
        Store(out, i, Load(out, i - 1) + seg);
        if (segLength > 1e-12f) {
            GfVec3f const unit = strand::Normalize(seg, dir, 1e-12f);
            dir = haveDir ? strand::Normalize(dir * 0.5f + unit * 0.5f, unit, 1e-12f) : unit;
            haveDir = true;
        }
    }
}

// ---------------------------------------------------------------------------
// Overrides and sub-clumps
// ---------------------------------------------------------------------------

GfVec3f SampleAtParameter(float const *p, size_t m, size_t i, size_t n)
{
    if (m == 0) return GfVec3f(0.0f);
    if (m == n) return Load(p, i);
    float const t = n > 1 ? float(i) / float(n - 1) : 0.0f;
    float const at = m > 1 ? t * float(m - 1) : 0.0f;
    size_t const lo = std::min(size_t(at), m - 1);
    size_t const hi = std::min(lo + 1, m - 1);
    float const f = at - float(lo);
    GfVec3f const a = Load(p, lo), b = Load(p, hi);
    return a + (b - a) * f;
}

namespace {

inline void Pull(float *p, GfVec3f const &target, float w)
{
    float const keep = 1.0f - w;
    p[0] = keep * p[0] + w * target[0];
    p[1] = keep * p[1] + w * target[1];
    p[2] = keep * p[2] + w * target[2];
}

}  // namespace

void ApplyOverrides(std::vector<uint32_t> const &A, std::vector<int> const &strandIndex,
                    VtIntArray const &overrides, float const *roots,
                    std::vector<float> const &snapshot, std::vector<float> *points,
                    std::vector<uint32_t> const &spans, std::vector<int32_t> *clumpId,
                    std::vector<char> const &flyaway, float flyawayAmount,
                    WeightFn const &weight)
{
    float const a = Clamp01(flyawayAmount);
    std::vector<float> &P = *points;
    // keep = 0 restores the snapshot; otherwise lerp(snapshot, P, keep).
    auto restore = [&](uint32_t c, float keep) {
        for (size_t k = 3 * size_t(spans[c]); k < 3 * size_t(spans[c + 1]); ++k)
            P[k] = keep <= 0.0f ? snapshot[k] : (1.0f - keep) * snapshot[k] + keep * P[k];
    };
    std::map<int, std::vector<uint32_t>> custom;
    for (uint32_t c : A) {
        int const s = strandIndex[c];
        if (s < 0 || size_t(s) >= overrides.size()) continue;
        int const v = overrides[size_t(s)];
        if (v == INT_MIN) continue;
        if (v == -1) {
            if (a <= 1e-6f) restore(c, 0.0f);
            else if (a < 0.999f) restore(c, a);
            (*clumpId)[c] = -1;
        } else if (v >= 0) {
            custom[v].push_back(c);
            (*clumpId)[c] = v;
        }
    }
    for (auto const &entry : custom) {
        std::vector<uint32_t> const &members = entry.second;
        if (members.size() < 2) continue;
        for (uint32_t c : members) restore(c, 0.0f);
        GfVec3f mean(0.0f);
        for (uint32_t c : members) mean += Load(roots, c);
        mean /= float(members.size());
        uint32_t centre = members.front();
        float best = FLT_MAX;
        for (uint32_t c : members) {
            GfVec3f const d = Load(roots, c) - mean;
            float const d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if (d2 < best) {
                best = d2;
                centre = c;
            }
        }
        float const *cp = &snapshot[3 * size_t(spans[centre])];
        size_t const cm = spans[centre + 1] - spans[centre];
        for (uint32_t c : members) {
            size_t const first = spans[c], n = spans[c + 1] - spans[c];
            for (size_t i = 0; i < n; ++i) {
                float w = weight(c, i, n);
                if (flyaway[c]) w *= a;
                if (w > 1e-6f) Pull(&P[3 * (first + i)], SampleAtParameter(cp, cm, i, n), w);
            }
        }
    }
}

void ApplySubClumps(std::vector<uint32_t> const &A, float const *liveRoots,
                    std::vector<float> *points, std::vector<uint32_t> const &spans,
                    std::vector<int32_t> *clumpId, std::vector<char> const &flyaway,
                    SubClumpSettings const &settings, WeightFn const &weight,
                    strand::TriangleBvh const *surface)
{
    int const S = std::max(1, settings.count);
    if (S <= 1) return;
    float const spreadT = Clamp01(settings.spread);
    float const amt = std::min(std::max(settings.amount, 0.1f), 1.0f);
    std::vector<float> &P = *points;

    std::map<int32_t, std::vector<uint32_t>> groups;
    for (uint32_t c : A)
        if (!flyaway[c] && (*clumpId)[c] >= 0) groups[(*clumpId)[c]].push_back(c);

    std::vector<std::pair<float, size_t>> angles;
    std::vector<uint32_t> sub;
    std::vector<float> curve;
    for (auto const &entry : groups) {
        int32_t const key = entry.first;
        std::vector<uint32_t> const &members = entry.second;
        size_t const m = members.size();
        if (m < 2) continue;
        size_t const Sg = std::min(size_t(S), m);

        GfVec3f centre(0.0f);
        for (uint32_t k : members) centre += Load(liveRoots, k);
        centre /= float(m);
        float sumSq = 0.0f;
        for (uint32_t k : members) sumSq += (Load(liveRoots, k) - centre).GetLengthSq();
        float const rms = std::sqrt(sumSq / float(m));
        float const spread = std::max(1e-4f, (0.15f + (2.7f - 0.15f) * spreadT) * rms);

        uint32_t lead = members.front();
        float leadD = FLT_MAX;
        for (uint32_t k : members) {
            float const d = (Load(liveRoots, k) - centre).GetLengthSq();
            if (d < leadD) {
                leadD = d;
                lead = k;
            }
        }
        size_t const ln = spans[lead + 1] - spans[lead];
        if (ln < 1) continue;
        float const *lp = &P[3 * size_t(spans[lead])];
        GfVec3f d = Load(lp, ln - 1) - Load(lp, 0);
        float const dl = d.GetLength();
        if (!(dl > 1e-6f)) continue;
        d /= dl;
        GfVec3f const ref = std::fabs(d[1]) >= 0.95f ? GfVec3f(1.0f, 0.0f, 0.0f)
                                                     : GfVec3f(0.0f, 1.0f, 0.0f);
        GfVec3f const e1 = strand::Normalize(GfCross(d, ref), GfVec3f(1.0f, 0.0f, 0.0f));
        GfVec3f const e2 = strand::Normalize(GfCross(d, e1), GfVec3f(0.0f, 0.0f, 1.0f));

        angles.clear();
        for (size_t k = 0; k < m; ++k) {
            GfVec3f const v = Load(liveRoots, members[k]) - centre;
            angles.emplace_back(std::atan2(GfDot(v, e2), GfDot(v, e1)), k);
        }
        std::stable_sort(angles.begin(), angles.end(),
                         [](auto const &a, auto const &b) { return a.first < b.first; });
        size_t start = 0;
        float bestGap = -1.0f;
        for (size_t i = 0; i < m; ++i) {
            float const gap = i + 1 < m ? angles[i + 1].first - angles[i].first
                                        : angles[0].first + kTwoPi - angles[m - 1].first;
            if (gap > bestGap) {
                bestGap = gap;
                start = (i + 1) % m;
            }
        }

        size_t offset = 0;
        for (size_t j = 0; j < Sg; ++j) {
            size_t const count = m / Sg + (j < m % Sg ? 1 : 0);
            sub.clear();
            for (size_t q = 0; q < count; ++q)
                sub.push_back(members[angles[(start + offset + q) % m].second]);
            offset += count;
            if (sub.empty()) continue;

            float sx = 0.0f, sy = 0.0f;
            for (uint32_t k : sub) {
                GfVec3f const v = Load(liveRoots, k) - centre;
                float const x = GfDot(v, e1), y = GfDot(v, e2);
                float const l = std::sqrt(x * x + y * y);
                if (l > 1e-12f) {
                    sx += x / l;
                    sy += y / l;
                }
            }
            float const phi = sx * sx + sy * sy > 1e-12f
                ? std::atan2(sy, sx) : float(j) / float(Sg) * kTwoPi;
            GfVec3f const o = (e1 * std::cos(phi) + e2 * std::sin(phi)) * spread;

            // Mean curve of the sector, at the first member's CV count.
            size_t const cn = spans[sub.front() + 1] - spans[sub.front()];
            if (cn == 0) continue;
            curve.assign(3 * cn, 0.0f);
            for (uint32_t k : sub) {
                size_t const kn = spans[k + 1] - spans[k];
                float const *kp = &P[3 * size_t(spans[k])];
                for (size_t i = 0; i < cn; ++i) {
                    GfVec3f const q = SampleAtParameter(kp, kn, i, cn);
                    curve[3 * i] += q[0];
                    curve[3 * i + 1] += q[1];
                    curve[3 * i + 2] += q[2];
                }
            }
            float const inv = 1.0f / float(sub.size());
            for (float &v : curve) v *= inv;
            GfVec3f const T = centre + o;
            GfVec3f landed = T;
            if (surface && !surface->Empty()) {
                strand::ClosestHit const hit = surface->Closest(T);
                if (hit.hit) landed = hit.point;
            }
            GfVec3f const c0 = Load(curve.data(), 0);
            for (size_t i = 0; i < cn; ++i) {
                float const u = cn > 1 ? float(i) / float(cn - 1) : 0.0f;
                Store(curve.data(), i,
                      Load(curve.data(), i) + (landed - c0) * (1.0f - u) + (T - c0) * u);
            }

            for (uint32_t k : sub) {
                size_t const first = spans[k], n = spans[k + 1] - spans[k];
                for (size_t i = 0; i < n; ++i) {
                    float const w = weight(k, i, n) * amt;
                    if (w > 1e-6f)
                        Pull(&P[3 * (first + i)], SampleAtParameter(curve.data(), cn, i, n), w);
                }
                if (settings.ids)
                    (*clumpId)[k] = int32_t(int64_t(key) * int64_t(S) + int64_t(j));
            }
        }
    }
}

}  // namespace clumpRef
}  // namespace usdGen
