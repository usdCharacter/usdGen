// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
//
// usdGen — randomRange length math. See lengthRange.h.
#include "usdGen/ops/lengthRange.h"

#include "usdGen/ops/strandUtil.h"
#include "usdGen/scheduler.h"

#include <algorithm>
#include <cmath>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {
namespace lengthRange {

namespace {

template <class F>
void ParallelFor(UsdGenWorkDispatcher *dispatcher, size_t count, F const &body)
{
    if (!dispatcher || count < 256) {
        for (size_t i = 0; i < count; ++i) body(i);
        return;
    }
    struct Payload { F const *body; } payload{&body};
    dispatcher->ParallelFor(count, [](size_t i, void *p) {
        (*static_cast<Payload *>(p)->body)(i);
    }, &payload);
}

/// One clump's mean curve: M[k] = (sum of member points k) * (1 / count_k),
/// summed in generation order.
struct MeanCurve
{
    std::vector<float> points;  // 3 * n
    std::vector<float> cum;
    size_t n = 0;
    float length = 0.0f;
    GfVec3f tip{0.0f, 1.0f, 0.0f};
};

}  // namespace

float RandomScale(Settings const &settings, uint32_t seed)
{
    float lo = std::min(std::max(settings.min, 0.01f), 2.0f);
    float hi = std::min(std::max(settings.max, 0.01f), 2.0f);
    if (hi < lo) std::swap(lo, hi);
    float const r = std::min(std::max(settings.randomize, 0.0f), 1.0f);
    float scale = hi;
    if (r > 0.0f && hi - lo > 1e-6f) {
        // The 32-bit key is sign-extended before hashing (int64(int32)).
        float const u = strand::Hash01(static_cast<uint64_t>(
            static_cast<int64_t>(static_cast<int32_t>(seed))));
        scale = (lo + u * (hi - lo)) * r + (1.0f - r) * hi;
    }
    return scale;
}

GfVec3f SampleForward(float const *R, size_t nR, std::vector<float> const &cum,
                      float L, GfVec3f const &tip, float x, size_t *k)
{
    if (nR == 0) return GfVec3f(0.0f);
    if (x < L && nR >= 2) {
        size_t kk = k ? *k : 0;
        if (kk + 1 >= nR) kk = nR - 2;
        while (kk + 2 < nR && cum[kk + 1] < x) ++kk;
        if (k) *k = kk;
        float const seg = cum[kk + 1] - cum[kk];
        float const w = seg > 1e-6f ? (x - cum[kk]) / seg : 0.0f;
        GfVec3f const a = strand::Load(R, kk), b = strand::Load(R, kk + 1);
        return a + (b - a) * w;
    }
    return strand::Load(R, nR - 1) + tip * (x - L);
}

void ResampleTo(float const *R, size_t nR, std::vector<float> const &cum, float L,
                GfVec3f const &tip, float target, size_t n, float *dst)
{
    size_t k = 0;
    for (size_t j = 0; j < n; ++j) {
        float const x = n > 1 ? float(j) / float(n - 1) * target : 0.0f;
        strand::Store(dst, j, SampleForward(R, nR, cum, L, tip, x, &k));
    }
}

GfVec3f TipDirection(float const *R, size_t n)
{
    if (n < 2) return GfVec3f(0.0f, 1.0f, 0.0f);
    return strand::Normalize(strand::Load(R, n - 1) - strand::Load(R, n - 2),
                             GfVec3f(0.0f, 1.0f, 0.0f), 1e-12f);
}

void Apply(Settings const &settings, Strands const &strands, std::vector<float> *points,
           std::vector<float> *restLength, UsdGenWorkDispatcher *dispatcher)
{
    if (!strands.spans || strands.spans->size() < 2 || !points || !restLength) return;
    std::vector<uint32_t> const &spans = *strands.spans;
    size_t const R = spans.size() - 1;
    restLength->assign(R, 0.0f);
    auto clumpOf = [&](size_t c) -> int32_t {
        return c < strands.clumpId.size() ? strands.clumpId[c] : -1;
    };

    // Clump mean curves (only consumed by followClumpMean). Ids are compacted
    // by sorting, so any id range works; every listed id has >= 1 member.
    std::vector<int32_t> ids;
    std::vector<MeanCurve> means;
    if (settings.followClumpMean) {
        for (size_t c = 0; c < R; ++c)
            if (clumpOf(c) >= 0 && spans[c + 1] > spans[c]) ids.push_back(clumpOf(c));
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
        means.resize(ids.size());
        auto groupOf = [&](int32_t cid) -> size_t {
            return size_t(std::lower_bound(ids.begin(), ids.end(), cid) - ids.begin());
        };
        for (size_t c = 0; c < R; ++c) {
            if (clumpOf(c) < 0) continue;
            MeanCurve &m = means[groupOf(clumpOf(c))];
            m.n = std::max(m.n, size_t(spans[c + 1] - spans[c]));
        }
        std::vector<std::vector<uint32_t>> counts(means.size());
        for (size_t g = 0; g < means.size(); ++g) {
            means[g].points.assign(3 * means[g].n, 0.0f);
            counts[g].assign(means[g].n, 0u);
        }
        bool const ordered = strands.order.size() == R;
        for (size_t i = 0; i < R; ++i) {
            size_t const c = ordered ? strands.order[i] : i;
            if (c >= R || clumpOf(c) < 0) continue;
            size_t const g = groupOf(clumpOf(c));
            size_t const n = spans[c + 1] - spans[c];
            float const *P = points->data() + 3 * size_t(spans[c]);
            for (size_t k = 0; k < n; ++k) {
                means[g].points[3 * k] += P[3 * k];
                means[g].points[3 * k + 1] += P[3 * k + 1];
                means[g].points[3 * k + 2] += P[3 * k + 2];
                ++counts[g][k];
            }
        }
        for (size_t g = 0; g < means.size(); ++g) {
            MeanCurve &m = means[g];
            for (size_t k = 0; k < m.n; ++k) {
                float const inv = 1.0f / float(std::max(counts[g][k], 1u));
                m.points[3 * k] *= inv;
                m.points[3 * k + 1] *= inv;
                m.points[3 * k + 2] *= inv;
            }
            m.length = strand::ArcLengths(m.points.data(), m.n, &m.cum);
            m.tip = TipDirection(m.points.data(), m.n);
        }
    }

    ParallelFor(dispatcher, R, [&](size_t c) {
        size_t const first = spans[c], n = size_t(spans[c + 1] - spans[c]);
        if (n == 0) return;
        float *P = points->data() + 3 * first;
        int32_t const cid = clumpOf(c);
        uint32_t const key = (settings.shareByClump && cid >= 0)
            ? uint32_t(cid) : (c < strands.seed.size() ? strands.seed[c] : uint32_t(c));
        float scale = RandomScale(settings, key);
        float const lm = c < strands.mask.size() ? strands.mask[c] : 1.0f;
        scale *= std::min(std::max(std::isfinite(lm) ? lm : 0.0f, 0.0f), 1.0f);

        MeanCurve const *mean = nullptr;
        if (settings.followClumpMean && cid >= 0) {
            auto const it = std::lower_bound(ids.begin(), ids.end(), cid);
            if (it != ids.end() && *it == cid && means[size_t(it - ids.begin())].n > 0)
                mean = &means[size_t(it - ids.begin())];
        }
        thread_local std::vector<float> original, cum;
        original.assign(P, P + 3 * n);
        float const *ref = original.data();
        size_t nRef = n;
        std::vector<float> const *cumRef = &cum;
        float L = 0.0f;
        GfVec3f tip(0.0f, 1.0f, 0.0f);
        if (mean) {
            ref = mean->points.data();
            nRef = mean->n;
            cumRef = &mean->cum;
            L = mean->length;
            tip = mean->tip;
        } else {
            L = strand::ArcLengths(original.data(), n, &cum);
            tip = TipDirection(original.data(), n);
        }
        (*restLength)[c] = L;
        if (!(L >= 1e-6f)) return;   // degenerate reference: strand unchanged
        float const target = L * scale;
        size_t k = 0;
        for (size_t j = 0; j < n; ++j) {
            float const x = n > 1 ? float(j) / float(n - 1) * target : 0.0f;
            GfVec3f q = SampleForward(ref, nRef, *cumRef, L, tip, x, &k);
            if (mean)
                q += strand::Load(original.data(), j) -
                     strand::Load(mean->points.data(), std::min(j, mean->n - 1));
            strand::Store(P, j, q);
        }
    });
}

}  // namespace lengthRange
}  // namespace usdGen
