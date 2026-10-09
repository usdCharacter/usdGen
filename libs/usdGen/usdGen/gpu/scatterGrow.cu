#include "scatterGrow.h"
#include "cudaCompat.h"

#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace usdGen::gpu {
namespace {
constexpr int kNonFinite = 2;
constexpr int kPendingStatus = std::numeric_limits<int>::min();
ScatterGrowStatus Status(cudaError_t e) { return e == cudaSuccess ? ScatterGrowStatus::Ok : ScatterGrowStatus::CudaError; }
bool Finite(double v) { return std::isfinite(v); }
bool Finite(float v) { return std::isfinite(v); }
bool Finite(float2 v) { return Finite(v.x) && Finite(v.y); }
bool Finite(float3 v) { return Finite(v.x) && Finite(v.y) && Finite(v.z); }

__device__ uint64_t Hash64(uint64_t key, uint32_t salt) {
    uint64_t z = (key ^ (uint64_t(salt) * 0x9E3779B97F4A7C15ull)) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ float DrawGrow(int seed, uint64_t id, uint32_t salt = 0x47726F77u) {
    uint64_t key = Hash64(uint64_t(uint32_t(seed)), salt) ^ id;
    return float(uint32_t(Hash64(key, salt) >> 32) >> 8) * 0x1.0p-24f;
}
__device__ float3 Normalize(float3 v) {
    float l2 = v.x*v.x + v.y*v.y + v.z*v.z;
    // Match CPU Grow's authored-direction fallback.
    float length = sqrtf(l2);
    if (!(length > 1.0e-12f) || !isfinite(length)) return make_float3(0, 1, 0);
    return make_float3(v.x/length, v.y/length, v.z/length);
}
__device__ float3 RotateAroundB(float3 direction, float3 axis, float degrees) {
    if (degrees == 0.0f) return direction;
    axis = Normalize(axis);
    constexpr float pi = 3.14159265358979323846f;
    float const radians = degrees * (pi / 180.0f);
    float const c = cosf(radians), s = sinf(radians);
    float const dot = axis.x * direction.x + axis.y * direction.y + axis.z * direction.z;
    float3 const cross = make_float3(
        axis.y * direction.z - axis.z * direction.y,
        axis.z * direction.x - axis.x * direction.z,
        axis.x * direction.y - axis.y * direction.x);
    float const oneMinusC = 1.0f - c;
    return make_float3(
        direction.x * c + cross.x * s + axis.x * dot * oneMinusC,
        direction.y * c + cross.y * s + axis.y * dot * oneMinusC,
        direction.z * c + cross.z * s + axis.z * dot * oneMinusC);
}
__device__ bool FiniteDevice(float x) { return isfinite(x); }
__device__ bool FiniteDevice(float3 v) {
    return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}
uint64_t Hash64Host(uint64_t key, uint32_t salt) {
    uint64_t z = (key ^ (uint64_t(salt) * 0x9E3779B97F4A7C15ull)) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
float DrawGrowHost(int seed, uint64_t id, uint32_t salt = 0x47726F77u) {
    uint64_t key = Hash64Host(uint64_t(uint32_t(seed)), salt) ^ id;
    return float(uint32_t(Hash64Host(key, salt) >> 32) >> 8) * 0x1.0p-24f;
}
// DrawGrowHost with the loop-invariant seed fold precomputed: keying by
// (seedHash ^ id) is exactly DrawGrowHost's spelling with the hoisted
// Hash64Host(seed32, salt). Integer-only; bit-identical by construction.
float DrawGrowHostSeeded(uint64_t seedHash, uint64_t id, uint32_t salt) {
    return float(uint32_t(Hash64Host(seedHash ^ id, salt) >> 32) >> 8) *
           0x1.0p-24f;
}
float3 NormalizeHost(float3 v) {
    float const l2 = v.x*v.x + v.y*v.y + v.z*v.z;
    float const length = std::sqrt(l2);
    if (!(length > 1.0e-12f) || !Finite(length)) return make_float3(0, 1, 0);
    return make_float3(v.x/length, v.y/length, v.z/length);
}
// Minimum second-occurrence index of any duplicated stable id (n when the
// ids are unique): the duplicate leg of the minima-combine validator.
// LSD radix over the 32-bit fold id^(id>>32) with a 32-bit satellite
// index is stable, so equal folds land in ascending original-index
// order. Groups of size 1 cannot duplicate; larger groups resolve
// against the full 64-bit ids (fold collisions are not duplicates), so
// the result is exact for any input. Skipping constant digits is exact
// (a constant-digit counting pass is the identity permutation) and
// keeps the surviving passes least- to most-significant. Two passes
// over packed 8-byte (fold,index) items beat the fused loop's 36MB
// random set probes and the 4-pass 12-byte full-id sort alike; packing
// halves the scatter's permuted stores (one 8-byte line hit instead of
// two 4-byte hits) for the same permutation. n <= UINT32_MAX (indices
// are 32-bit); the caller routes larger inputs to SetFirstDup.
// Pooled dup-sort scratch (bit-identical): the sharded path's items/tmp
// (8MB each at 1M roots) + histograms (2MB) and the legacy path's
// items/tmp + counts/par tables churn fresh alloc+fault+free every
// ValidateRoots call. The sharded and legacy paths never run in one
// call, so one item pair plus one table block serves both. Growth-only
// (never shrinks); the pack/partition writes every item slot, every
// radix pass fully writes its output before the swap, and every table
// is filled/prefixed before its read, so pooled contents never leak.
// (An earlier round pooled only the par tables and measured slower --
// fresh allocs then recycled hot pages. With the capture/sort/shell
// pools retaining ~270MB/thread there are no hot bins left, so pooling
// the whole temp set is what removes the fresh faults.)
struct DupSortScratch {
    std::unique_ptr<uint64_t[]> items;
    std::unique_ptr<uint64_t[]> tmp;
    std::unique_ptr<uint32_t[]> tab;
    size_t nMax = 0;
};
thread_local DupSortScratch t_dupSortScratch;
size_t RadixFirstDup(uint64_t const* ids, size_t n) {
    if (n <= 1) return n;
    DupSortScratch& ds = t_dupSortScratch;
    if (!ds.tab)
        ds.tab.reset(new uint32_t[size_t(2) * 8 * 65536]);
    if (ds.nMax < n) {
        ds.items.reset(new uint64_t[n]);
        ds.tmp.reset(new uint64_t[n]);
        ds.nMax = n;
    }
    {
        // Sharded dup sort (verdict-identical): hash-fold digits scatter
        // uniformly at random, so the global passes below dirty one 64B
        // line per 8B item (~8x write amplification over the 8MB output,
        // ~3.4ms of cuda_validate). A stable 8-way partition by fold
        // low-3-bits runs first (8 cursors per chunk: sequential runs),
        // then each shard sorts serially on its worker with the same
        // (fold,index) packing, the same stable radix passes, and the
        // same per-group resolve below; the working set per shard
        // (~2MB) stays cache-contained instead of spilling. Equal folds
        // share their low 3 bits, so no fold group ever splits across
        // shards; chunk ranges ascend, so each shard's partition order
        // is global input order and every shard resolves the same
        // second-occurrences the global sort would. The global minimum
        // over shards is the identical index. A pathological skew (one
        // shard past n/2) falls through to the legacy global path.
        int const shardWorkers = tbb::this_task_arena::max_concurrency();
        size_t const shardChunks =
            (shardWorkers > 1 && n > 32768)
                ? std::min({size_t(shardWorkers), size_t(8), n})
                : 1;
        if (shardChunks > 1) {
            constexpr size_t kShards = 8;
            static_assert((kShards & (kShards - 1)) == 0,
                          "shard mask needs a power of two");
            constexpr size_t kMaxChunks = 8;
            size_t chunkShard[kMaxChunks][kShards];
            for (size_t c = 0; c < kMaxChunks; ++c)
                for (size_t s = 0; s < kShards; ++s)
                    chunkShard[c][s] = 0;
            // Hoisted fold or/and (bit-identical): the count loop below
            // already folds every id, so it also accumulates the global
            // fold |/& reduction (order-free) into per-chunk partials.
            // The per-shard digit-skip then reads the global vary instead
            // of re-streaming each shard's items: a globally-constant
            // digit is constant in every shard, so the same passes skip.
            // A globally-varying digit runs in every shard (even a
            // constant one: correct, just work, only on dup-heavy
            // inputs); the sorted arrays are identical either way.
            uint32_t chunkOr[kMaxChunks] = {};
            uint32_t chunkAnd[kMaxChunks];
            for (size_t c = 0; c < kMaxChunks; ++c)
                chunkAnd[c] = ~uint32_t(0);
            uint64_t const* idsIn = ids;
            size_t const nn = n;
            size_t const nch = shardChunks;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, nch),
                [&](tbb::blocked_range<size_t> const& range) {
                    for (size_t c = range.begin(); c != range.end(); ++c) {
                        size_t const i0 = (c * nn) / nch;
                        size_t const i1 = ((c + 1) * nn) / nch;
                        size_t local[kShards] = {};
                        uint32_t orLocal = 0, andLocal = ~uint32_t(0);
                        for (size_t i = i0; i < i1; ++i) {
                            uint64_t const id = idsIn[i];
                            uint32_t const fold =
                                uint32_t(id) ^ uint32_t(id >> 32);
                            ++local[fold & (kShards - 1)];
                            orLocal |= fold;
                            andLocal &= fold;
                        }
                        for (size_t s = 0; s < kShards; ++s)
                            chunkShard[c][s] = local[s];
                        chunkOr[c] = orLocal;
                        chunkAnd[c] = andLocal;
                    }
                });
            uint32_t shardOr = 0, shardAnd = ~uint32_t(0);
            for (size_t c = 0; c < nch; ++c) {
                shardOr |= chunkOr[c];
                shardAnd &= chunkAnd[c];
            }
            uint32_t const shardVary = shardOr ^ shardAnd;
            size_t shardBase[kShards + 1];
            size_t chunkCur[kMaxChunks][kShards];
            shardBase[0] = 0;
            for (size_t s = 0; s < kShards; ++s) {
                size_t base = shardBase[s];
                for (size_t c = 0; c < nch; ++c) {
                    chunkCur[c][s] = base;
                    base += chunkShard[c][s];
                }
                shardBase[s + 1] = base;
            }
            size_t maxShard = 0;
            for (size_t s = 0; s < kShards; ++s) {
                size_t const m = shardBase[s + 1] - shardBase[s];
                if (m > maxShard) maxShard = m;
            }
            if (maxShard <= n / 2) {
                // Pooled (see above): the shard arrays alias the scratch.
                uint64_t* shardItems = ds.items.get();
                uint64_t* shardTmp = ds.tmp.get();
                uint32_t* shardHist = ds.tab.get();
                uint64_t* partOut = shardItems;
                tbb::parallel_for(tbb::blocked_range<size_t>(0, nch),
                    [&](tbb::blocked_range<size_t> const& range) {
                        for (size_t c = range.begin(); c != range.end();
                             ++c) {
                            size_t const i0 = (c * nn) / nch;
                            size_t const i1 = ((c + 1) * nn) / nch;
                            size_t cur[kShards];
                            for (size_t s = 0; s < kShards; ++s)
                                cur[s] = chunkCur[c][s];
                            for (size_t i = i0; i < i1; ++i) {
                                uint64_t const id = idsIn[i];
                                uint32_t const fold =
                                    uint32_t(id) ^ uint32_t(id >> 32);
                                size_t const s = fold & (kShards - 1);
                                partOut[cur[s]++] =
                                    (uint64_t(fold) << 32) | uint32_t(i);
                            }
                        }
                    });
                // Per-group second-occurrence resolution, shared logic
                // with the legacy path below: members run..i-1 share a
                // fold in ascending-index order. Returns the minimum of
                // dBad and the group's second occurrences.
                auto shardResolveGroup = [&](uint64_t const* sorted,
                                             size_t run, size_t i,
                                             std::vector<uint32_t>& big,
                                             size_t dBad) -> size_t {
                    size_t const g = i - run;
                    if (g > 64) {
                        big.resize(g);
                        for (size_t t = 0; t < g; ++t)
                            big[t] = uint32_t(sorted[run + t]);
                        std::sort(big.begin(), big.end(),
                            [&](uint32_t a, uint32_t b) {
                                return ids[a] != ids[b] ? ids[a] < ids[b]
                                                       : a < b;
                            });
                        for (size_t j = 1; j < g; ++j) {
                            if (ids[big[j]] != ids[big[j - 1]]) continue;
                            if (j > 1 &&
                                ids[big[j - 1]] == ids[big[j - 2]])
                                continue;
                            if (size_t(big[j]) < dBad) {
                                dBad = size_t(big[j]);
                                if (dBad == 1) return 1;
                            }
                        }
                    } else {
                        for (size_t j = run + 1; j < i; ++j) {
                            uint32_t const jj = uint32_t(sorted[j]);
                            int matches = 0;
                            for (size_t q = run; q < j; ++q) {
                                if (ids[jj] ==
                                        ids[uint32_t(sorted[q])] &&
                                    ++matches > 1)
                                    break;
                            }
                            if (matches == 1 && size_t(jj) < dBad) {
                                dBad = size_t(jj);
                                if (dBad == 1) return 1;
                            }
                        }
                    }
                    return dBad;
                };
                std::vector<size_t> shardMin(kShards, n);
                tbb::parallel_for(tbb::blocked_range<size_t>(0, kShards),
                    [&](tbb::blocked_range<size_t> const& range) {
                        for (size_t s = range.begin(); s != range.end();
                             ++s) {
                            size_t const b0 = shardBase[s];
                            size_t const m = shardBase[s + 1] - b0;
                            if (m <= 1) continue;
                            uint64_t* w = shardItems + b0;
                            uint64_t* wOut = shardTmp + b0;
                            uint32_t* cnt =
                                shardHist + s * 65536;
                            // Hoisted global vary (see the count loop):
                            // a globally-constant digit skips in every
                            // shard, the same passes today's prescan
                            // skipped; the sorted shards are identical.
                            uint32_t const vary = shardVary;
                            for (int pass = 0; pass < 2; ++pass) {
                                if (((vary >> (pass * 16)) & 0xffffu) == 0)
                                    continue;
                                int const shift = 32 + pass * 16;
                                std::fill(cnt, cnt + 65536, uint32_t(0));
                                for (size_t k = 0; k < m; ++k)
                                    ++cnt[uint32_t(w[k] >> shift) &
                                           0xffffu];
                                uint32_t sum = 0;
                                for (size_t c = 0; c < 65536; ++c) {
                                    uint32_t const t = cnt[c];
                                    cnt[c] = sum;
                                    sum += t;
                                }
                                for (size_t k = 0; k < m; ++k) {
                                    uint64_t const wk = w[k];
                                    size_t const d =
                                        (uint32_t(wk >> shift)) & 0xffffu;
                                    wOut[cnt[d]++] = wk;
                                }
                                std::swap(w, wOut);
                            }
                            size_t local = n;
                            std::vector<uint32_t> big;
                            size_t run = 0;
                            for (size_t j = 1; j <= m; ++j) {
                                if (j < m &&
                                    uint32_t(w[j] >> 32) ==
                                        uint32_t(w[run] >> 32))
                                    continue;
                                if (j - run > 1) {
                                    local = shardResolveGroup(w, run, j,
                                                              big, local);
                                    if (local == 1) break;
                                }
                                run = j;
                            }
                            shardMin[s] = local;
                        }
                    });
                size_t dBad = n;
                for (size_t s = 0; s < kShards; ++s)
                    dBad = shardMin[s] < dBad ? shardMin[s] : dBad;
                return dBad;
            }
        }
    }
    // Uninitialized scratch (bit-identical): every slot of both item
    // arrays is overwritten before its read (pack fill, radix passes),
    // and counts is filled before each pass, so std::vector's value-init
    // (~16MB of zeroes) is pure waste; new[] leaves the trivial storage
    // uninitialized. Same bytes in the same slots. Pooled (see above):
    // the legacy arrays alias the same scratch the sharded path uses.
    uint64_t* items = ds.items.get();
    uint64_t* tmpItems = ds.tmp.get();
    uint32_t* counts = ds.tab.get();
    // Threaded pack + radix passes for big inputs (same n>32768 rule
    // the Capture sort uses; small inputs keep the serial spelling):
    // the pack writes disjoint slots with an order-free |/& reduction,
    // and each counting pass histograms per-chunk, prefixes once over
    // the chunk-major tables in chunk order, and scatters each chunk's
    // range in input order -- within a digit the landing order is chunk
    // order then input order, exactly the serial scatter's order, so
    // any chunking sorts bit-identically and the resolve below reads
    // the same array. Plain TBB (validate runs outside any arena).
    int const dupWorkers = tbb::this_task_arena::max_concurrency();
    size_t const dupChunks =
        (dupWorkers > 1 && n > 32768)
            ? std::min({size_t(dupWorkers), size_t(8), n})
            : 1;
    uint32_t orKeys = 0, andKeys = ~uint32_t(0);
    uint64_t* itemOut = items;
    if (dupChunks == 1) {
        for (size_t i = 0; i < n; ++i) {
            uint64_t const id = ids[i];
            uint32_t const key = uint32_t(id) ^ uint32_t(id >> 32);
            itemOut[i] = (uint64_t(key) << 32) | uint32_t(i);
            orKeys |= key;
            andKeys &= key;
        }
    } else {
        struct DupPackPartial { uint32_t orKeys = 0, andKeys = ~uint32_t(0); };
        std::vector<DupPackPartial> partial(dupChunks);
        tbb::parallel_for(tbb::blocked_range<size_t>(0, dupChunks),
            [&](tbb::blocked_range<size_t> const& range) {
                for (size_t c = range.begin(); c != range.end(); ++c) {
                    size_t const i0 = (c * n) / dupChunks;
                    size_t const i1 = ((c + 1) * n) / dupChunks;
                    uint32_t orLocal = 0, andLocal = ~uint32_t(0);
                    for (size_t i = i0; i < i1; ++i) {
                        uint64_t const id = ids[i];
                        uint32_t const key =
                            uint32_t(id) ^ uint32_t(id >> 32);
                        itemOut[i] = (uint64_t(key) << 32) | uint32_t(i);
                        orLocal |= key;
                        andLocal &= key;
                    }
                    partial[c].orKeys = orLocal;
                    partial[c].andKeys = andLocal;
                }
            });
        for (auto const& p : partial) {
            orKeys |= p.orKeys;
            andKeys &= p.andKeys;
        }
    }
    uint64_t* w = items;
    uint64_t* wOut = tmpItems;
    uint32_t* cnt = counts;
    uint32_t const vary = orKeys ^ andKeys;
    // Per-chunk histograms + scatter offsets for the threaded passes
    // (chunk-major [c * 65536 + d], each chunk's worker touching only
    // its own region). Pooled in the table block's two halves (dupChunks
    // <= 8, so need <= half a block each). Every counts slot is filled
    // each pass and every offset slot is prefixed, so no contents leak
    // anywhere. (Pooling only these tables measured slower when fresh
    // allocs recycled hot pages; with the capture pools retaining the
    // heap, pooling the whole temp set removes the fresh faults.)
    uint32_t* parCnt = nullptr;
    uint32_t* parOff = nullptr;
    if (dupChunks > 1) {
        parCnt = ds.tab.get();
        parOff = ds.tab.get() + size_t(8) * 65536;
    }
    for (int pass = 0; pass < 2; ++pass) {
        int const shift = 32 + pass * 16;
        if (((vary >> (pass * 16)) & 0xffffu) == 0) continue;
        if (dupChunks == 1) {
            std::fill(cnt, cnt + 65536, uint32_t(0));
            for (size_t i = 0; i < n; ++i)
                ++cnt[uint32_t(w[i] >> shift) & 0xffffu];
            uint32_t sum = 0;
            for (size_t c = 0; c < 65536; ++c) {
                uint32_t const t = cnt[c];
                cnt[c] = sum;
                sum += t;
            }
            for (size_t i = 0; i < n; ++i) {
                uint64_t const wi = w[i];
                size_t const d = (uint32_t(wi >> shift)) & 0xffffu;
                wOut[cnt[d]++] = wi;
            }
        } else {
            uint32_t* cntBase = parCnt;
            uint32_t* offBase = parOff;
            std::fill(cntBase, cntBase + dupChunks * 65536, uint32_t(0));
            uint64_t const* wIn = w;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, dupChunks),
                [&](tbb::blocked_range<size_t> const& range) {
                    for (size_t c = range.begin(); c != range.end(); ++c) {
                        size_t const i0 = (c * n) / dupChunks;
                        size_t const i1 = ((c + 1) * n) / dupChunks;
                        uint32_t* hc = cntBase + c * 65536;
                        for (size_t i = i0; i < i1; ++i)
                            ++hc[uint32_t(wIn[i] >> shift) & 0xffffu];
                    }
                });
            uint32_t sum = 0;
            for (size_t d = 0; d < 65536; ++d) {
                for (size_t c = 0; c < dupChunks; ++c) {
                    size_t const s = c * 65536 + d;
                    uint32_t const t = cntBase[s];
                    offBase[s] = sum;
                    sum += t;
                }
            }
            uint64_t* oOut = wOut;
            tbb::parallel_for(tbb::blocked_range<size_t>(0, dupChunks),
                [&](tbb::blocked_range<size_t> const& range) {
                    for (size_t c = range.begin(); c != range.end(); ++c) {
                        size_t const i0 = (c * n) / dupChunks;
                        size_t const i1 = ((c + 1) * n) / dupChunks;
                        uint32_t* off = offBase + c * 65536;
                        for (size_t i = i0; i < i1; ++i) {
                            uint64_t const wi = wIn[i];
                            size_t const dd =
                                (uint32_t(wi >> shift)) & 0xffffu;
                            oOut[off[dd]++] = wi;
                        }
                    }
                });
        }
        std::swap(w, wOut);
    }
    // Per-group second-occurrence resolution, shared by the serial scan
    // and the parallel chunks below: members run..i-1 share a fold in
    // ascending-index order, so full-id loads stream rather than scatter.
    // Returns the minimum of dBad and the group's second occurrences (1
    // is the global floor: no second occurrence sits below index 1).
    uint64_t const *wSorted = w;
    auto foldAt = [wSorted](size_t i) -> uint32_t {
        return uint32_t(wSorted[i] >> 32);
    };
    auto resolveGroup = [&](size_t run, size_t i,
                            std::vector<uint32_t> &big,
                            size_t dBad) -> size_t {
        size_t const g = i - run;
        if (g > 64) {
            // Scratch for oversized fold groups (pathological shared
            // folds): sort the group's members by full id, then the
            // adjacent scan below is exact.
            big.resize(g);
            for (size_t t = 0; t < g; ++t)
                big[t] = uint32_t(wSorted[run + t]);
            std::sort(big.begin(), big.end(), [&](uint32_t a, uint32_t b) {
                return ids[a] != ids[b] ? ids[a] < ids[b] : a < b;
            });
            // Sorted by (full id, index): within an equal-id run the
            // indices ascend, so run[1] is that id's 2nd occurrence.
            for (size_t j = 1; j < g; ++j) {
                if (ids[big[j]] != ids[big[j - 1]]) continue;
                if (j > 1 && ids[big[j - 1]] == ids[big[j - 2]]) continue;
                if (size_t(big[j]) < dBad) {
                    dBad = size_t(big[j]);
                    if (dBad == 1) return 1;
                }
            }
        } else {
            for (size_t j = run + 1; j < i; ++j) {
                // Member j is a true second occurrence exactly when
                // one earlier member shares its full id.
                uint32_t const jj = uint32_t(wSorted[j]);
                int matches = 0;
                for (size_t q = run; q < j; ++q) {
                    if (ids[jj] == ids[uint32_t(wSorted[q])] && ++matches > 1)
                        break;
                }
                if (matches == 1 && size_t(jj) < dBad) {
                    dBad = size_t(jj);
                    if (dBad == 1) return 1;
                }
            }
        }
        return dBad;
    };
    // Parallel group resolution for big inputs (same rule as the radix
    // passes; small inputs keep the serial scan): each chunk resolves
    // the fold groups fully inside its range with the identical
    // per-group logic, skipping groups that touch either edge, and a
    // serial pass resolves each boundary-straddling group once
    // (boundaries inside an already-resolved group compare equal to its
    // start, so every group resolves exactly once). Every multi-group
    // resolves with the same logic either way and the minima combine
    // order-free, so any chunking returns the identical index.
    size_t dBad = n;
    if (dupChunks == 1) {
        size_t run = 0;
        std::vector<uint32_t> big;
        for (size_t i = 1; i <= n; ++i) {
            if (i < n && foldAt(i) == foldAt(run)) continue;
            size_t const g = i - run;
            if (g > 1) {
                dBad = resolveGroup(run, i, big, dBad);
                if (dBad == 1) return 1;
            }
            run = i;
        }
        return dBad;
    }
    std::vector<size_t> chunkMin(dupChunks, n);
    tbb::parallel_for(tbb::blocked_range<size_t>(0, dupChunks),
        [&](tbb::blocked_range<size_t> const& range) {
            for (size_t c = range.begin(); c != range.end(); ++c) {
                size_t const i0 = (c * n) / dupChunks;
                size_t const i1 = ((c + 1) * n) / dupChunks;
                // Owned sub-range: skip the left-partial group (it
                // extends below i0) and stop at the right-partial
                // group's start (it extends past i1). What remains
                // holds only complete groups.
                size_t s0 = i0, s1 = i1;
                if (c > 0 && s0 < n && foldAt(s0) == foldAt(s0 - 1))
                    while (s0 < n && foldAt(s0) == foldAt(i0)) ++s0;
                if (i1 < n && i1 > s0 && foldAt(i1 - 1) == foldAt(i1)) {
                    s1 = i1 - 1;
                    while (s1 > s0 && foldAt(s1 - 1) == foldAt(i1)) --s1;
                }
                size_t local = n;
                if (s1 > s0) {
                    std::vector<uint32_t> big;
                    size_t run = s0;
                    for (size_t j = s0 + 1; j <= s1; ++j) {
                        if (j < s1 && foldAt(j) == foldAt(run)) continue;
                        if (j - run > 1) {
                            local = resolveGroup(run, j, big, local);
                            if (local == 1) break;
                        }
                        run = j;
                    }
                }
                chunkMin[c] = local;
            }
        });
    for (size_t c = 0; c < dupChunks; ++c)
        dBad = chunkMin[c] < dBad ? chunkMin[c] : dBad;
    if (dBad > 1) {
        std::vector<uint32_t> big;
        size_t lastR = 0;
        for (size_t c = 0; c + 1 < dupChunks; ++c) {
            size_t const b = ((c + 1) * n) / dupChunks;
            if (b == 0 || b >= n || foldAt(b - 1) != foldAt(b)) continue;
            size_t L = b - 1;
            while (L > 0 && foldAt(L - 1) == foldAt(b)) --L;
            if (L < lastR) continue;  // inside an already-resolved group
            size_t R = b + 1;
            while (R < n && foldAt(R) == foldAt(b)) ++R;
            dBad = resolveGroup(L, R, big, dBad);
            lastR = R;
            if (dBad == 1) break;
        }
    }
    return dBad;
}
// Set-based first-duplicate leg for n > UINT32_MAX (no 32-bit index
// radix): insertion order is index order, so the first repeat found is
// the minimum second-occurrence index. Same open-addressed set the
// fused loop used.
size_t SetFirstDup(uint64_t const* ids, size_t n) {
    size_t setCap = 16;
    while (setCap <= n) setCap *= 2;
    setCap *= 2;
    std::vector<uint64_t> setKeys(setCap);
    std::vector<unsigned char> setUsed(setCap, 0);
    size_t const setMask = setCap - 1;
    for (size_t i = 0; i < n; ++i) {
        uint64_t const id = ids[i];
        size_t slot = size_t(Hash64Host(id, 0x9E3779B9u) & uint64_t(setMask));
        while (true) {
            if (!setUsed[slot]) {
                setUsed[slot] = 1;
                setKeys[slot] = id;
                break;
            }
            if (setKeys[slot] == id) return i;
            slot = (slot + 1) & setMask;
        }
    }
    return n;
}
float3 RotateAroundBHost(float3 direction, float3 axis, float degrees) {
    if (degrees == 0.0f) return direction;
    axis = NormalizeHost(axis);
    constexpr float pi = 3.14159265358979323846f;
    float const radians = degrees * (pi / 180.0f);
    float const c = std::cos(radians), s = std::sin(radians);
    float const dot = axis.x * direction.x + axis.y * direction.y + axis.z * direction.z;
    float3 const cross = make_float3(
        axis.y * direction.z - axis.z * direction.y,
        axis.z * direction.x - axis.x * direction.z,
        axis.x * direction.y - axis.y * direction.x);
    float const oneMinusC = 1.0f - c;
    return make_float3(
        direction.x * c + cross.x * s + axis.x * dot * oneMinusC,
        direction.y * c + cross.y * s + axis.y * dot * oneMinusC,
        direction.z * c + cross.z * s + axis.z * dot * oneMinusC);
}
__global__ void GrowKernel(float3 const* roots, uint64_t const* ids,
    float3 const* rootTIn,
    float3 const* rootBIn, float3 const* rootNIn, uint32_t curves, uint32_t cvCount,
    int seed, double length, double lo, double hi, float lift, float azimuth, float azimuthRandom, float width,
    ScatterGrowDirection direction, float3 literal,
    float3* points, float* widths, float* hairT, uint32_t* offsets,
    int* error) {
    uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= curves) return;
    offsets[c] = c * cvCount;
    if (c == curves - 1) offsets[curves] = curves * cvCount;
    float3 dir = direction == ScatterGrowDirection::RootNormal ? rootNIn[c] :
        direction == ScatterGrowDirection::RootTangent ? rootTIn[c] : literal;
    dir = Normalize(dir);
    dir = RotateAroundB(dir, rootBIn[c], lift);
    float const angle = azimuth + azimuthRandom * 360.0f *
        (DrawGrow(seed, ids[c], 0x4772417Au) - 0.5f); // kSaltGrowAzimuth
    dir = RotateAroundB(dir, rootNIn[c], angle);
    // Match CPU Grow: random/length arithmetic is double, then the captured
    // per-curve target is narrowed to float.
    double targetDouble = length * (lo + double(DrawGrow(seed, ids[c])) * (hi - lo));
    float target = float(targetDouble);
    if (!FiniteDevice(target)) { atomicCAS(error, 0, kNonFinite); return; }
    uint32_t first = c * cvCount;
    // cvCount divisible by 4 (validated 2..64): every curve's first point is
    // 16-byte aligned (first = c*cvCount, cudaMalloc is 256-byte aligned),
    // so four float3s pack into three float4 stores with identical bytes.
    // One 16-byte stream replaces four strided 12-byte streams; the uniform
    // branch keeps odd counts on the scalar spelling below.
    if ((cvCount & 3u) == 0) {
        float4* vpts = reinterpret_cast<float4*>(points + first);
        float3 r = roots[c];
        for (uint32_t i = 0; i < cvCount; i += 4) {
            float t0 = float(i) / float(cvCount - 1);
            float t1 = float(i + 1) / float(cvCount - 1);
            float t2 = float(i + 2) / float(cvCount - 1);
            float t3 = float(i + 3) / float(cvCount - 1);
            float d0 = target * t0, d1 = target * t1, d2 = target * t2, d3 = target * t3;
            if (!FiniteDevice(d0) || !FiniteDevice(d1) || !FiniteDevice(d2) || !FiniteDevice(d3)) { atomicCAS(error, 0, kNonFinite); return; }
            float3 p0 = make_float3(r.x + dir.x*d0, r.y + dir.y*d0, r.z + dir.z*d0);
            float3 p1 = make_float3(r.x + dir.x*d1, r.y + dir.y*d1, r.z + dir.z*d1);
            float3 p2 = make_float3(r.x + dir.x*d2, r.y + dir.y*d2, r.z + dir.z*d2);
            float3 p3 = make_float3(r.x + dir.x*d3, r.y + dir.y*d3, r.z + dir.z*d3);
            if (!FiniteDevice(p0) || !FiniteDevice(p1) || !FiniteDevice(p2) || !FiniteDevice(p3)) { atomicCAS(error, 0, kNonFinite); return; }
            vpts[0] = make_float4(p0.x, p0.y, p0.z, p1.x);
            vpts[1] = make_float4(p1.y, p1.z, p2.x, p2.y);
            vpts[2] = make_float4(p2.z, p3.x, p3.y, p3.z);
            vpts += 3;
            widths[first+i] = width; widths[first+i+1] = width; widths[first+i+2] = width; widths[first+i+3] = width;
            hairT[first+i] = t0; hairT[first+i+1] = t1; hairT[first+i+2] = t2; hairT[first+i+3] = t3;
        }
    } else {
        for (uint32_t i = 0; i < cvCount; ++i) {
            float t = float(i) / float(cvCount - 1);
            float d = target * t;
            if (!FiniteDevice(d)) { atomicCAS(error, 0, kNonFinite); return; }
            float3 p = make_float3(roots[c].x + dir.x*d, roots[c].y + dir.y*d, roots[c].z + dir.z*d);
            if (!FiniteDevice(p)) { atomicCAS(error, 0, kNonFinite); return; }
            points[first+i] = p; widths[first+i] = width; hairT[first+i] = t;
        }
    }
}

template <class T> cudaError_t Allocate(DeviceBuffer<T>& dst, std::vector<T> const& src,
                                         UsdGenExecutionMemoryReservation* r) { return dst.reset(src.size(), r); }
// Count twin for the VtArray-backed roots planes (Allocate only sizes).
template <class T> cudaError_t Allocate(DeviceBuffer<T>& dst, size_t n,
                                        UsdGenExecutionMemoryReservation* r) { return dst.reset(n, r); }
template <class T> cudaError_t Copy(DeviceBuffer<T>& dst, std::vector<T> const& src, cudaStream_t s) {
    return src.empty() ? cudaSuccess : cudaMemcpyAsync(dst.data(), src.data(), src.size()*sizeof(T), cudaMemcpyHostToDevice, s);
}
// Pointer/size twin for the VtArray-backed roots planes: same empty
// short-circuit (a null base with zero elements never reaches the copy),
// same bytes.
template <class T> cudaError_t Copy(DeviceBuffer<T>& dst, T const* src, size_t n, cudaStream_t s) {
    return n == 0 ? cudaSuccess : cudaMemcpyAsync(dst.data(), src, n*sizeof(T), cudaMemcpyHostToDevice, s);
}

// Pinned error-relay pool: BeginFresh allocates one pinned int per grow
// (~0.85ms cudaHostAlloc) and frees it at teardown. Proven-quiescent
// relays are retained process-wide (a few dozen 4-byte slots) and
// reissued, eliding the page-lock round-trip. Only proven paths return
// relays (unproven work still abandons in the destructor); every
// checkout re-reserves through the normal permit path, so accounting
// reads exactly as if freed. Deliberately leaked: entries must never
// run cudaFreeHost during process teardown.
std::mutex& PinnedRelayMutex() {
    static auto* m = new std::mutex;
    return *m;
}
std::vector<int*>& PinnedRelays() {
    static auto* v = new std::vector<int*>;
    return *v;
}
constexpr size_t kPinnedRelayCap = 64;
int* PopPinnedRelay() {
    std::lock_guard<std::mutex> lock(PinnedRelayMutex());
    auto& v = PinnedRelays();
    if (v.empty()) return nullptr;
    int* relay = v.back();
    v.pop_back();
    return relay;
}
void PushPinnedRelay(int* relay) {
    {
        std::lock_guard<std::mutex> lock(PinnedRelayMutex());
        auto& v = PinnedRelays();
        if (v.size() < kPinnedRelayCap) {
            v.push_back(relay);
            return;
        }
    }
    cudaFreeHost(relay);
}
// Cheap prefix of ValidateRoots (verdict-identical): argument checks,
// plane-size consistency, and the requirements arithmetic. The scan legs
// below it are the ~1.8ms; BeginFresh spec-allocates from this prefix and
// overlaps the H2D copies with the scans on a worker thread.
ScatterGrowStatus PrecheckRoots(
    std::shared_ptr<const ScatterGrowRoots> const& r, ScatterGrowControls const& c,
    size_t* n, size_t* total) {
    if(!r || c.cvCount<2 || c.cvCount>64 || !Finite(c.length)||!Finite(c.randomLo)||
       !Finite(c.randomHi)||!Finite(c.lift)||!Finite(c.fallbackWidth)||c.length<0||
       c.randomLo<0||c.randomHi<0||c.fallbackWidth<0 || c.lift < -90.0f ||
       c.lift > 90.0f || c.direction>ScatterGrowDirection::Literal ||
       !Finite(c.azimuth) || c.azimuth < -360.0f || c.azimuth > 360.0f ||
       !Finite(c.azimuthRandom) || c.azimuthRandom < 0.0f || c.azimuthRandom > 1.0f ||
       (c.direction==ScatterGrowDirection::Literal&&!Finite(c.literalDirection)))
        return ScatterGrowStatus::InvalidArgument;
    *n=r->positions.size();
    if(r->stableIds.size()!=*n||r->rootPrim.size()!=*n||r->rootUV.size()!=*n||
       r->rootT.size()!=*n||r->rootB.size()!=*n||r->rootN.size()!=*n)
        return ScatterGrowStatus::InvalidTopology;
    ScatterGrowRequirements requirements;
    auto requirementStatus = GetScatterGrowRequirements(*n, c.cvCount, &requirements);
    if (requirementStatus != ScatterGrowStatus::Ok) return requirementStatus;
    *total = requirements.pointCount;
    return ScatterGrowStatus::Ok;
}
} // namespace

ScatterGrowStatus GetScatterGrowRequirements(size_t curves, uint32_t cvs,
                                             ScatterGrowRequirements* result) {
    if (!result || cvs < 2 || cvs > 64) return ScatterGrowStatus::InvalidArgument;
    if (curves > std::numeric_limits<uint32_t>::max() / cvs)
        return ScatterGrowStatus::InvalidTopology;
    ScatterGrowRequirements candidate;
    candidate.pointCount = curves * cvs;
    // Input staging is root positions only: ids/prim/uv/frames upload
    // directly into their published output buffers (BeginFresh), so they
    // are counted in outputBytes below, not here.
    size_t const rootBytes = sizeof(float3);
    // One float3 per point: rest aliases the points buffer (see Storage),
    // so there is no second rest allocation to account.
    size_t const pointBytes = sizeof(float3) + 2 * sizeof(float);
    size_t const curveBytes = 3 * sizeof(float3) + sizeof(uint64_t) +
        sizeof(int32_t) + sizeof(float2) + sizeof(uint32_t);
    auto multiply = [](size_t a, size_t b, size_t* out) {
        if (a > std::numeric_limits<size_t>::max() / b) return false;
        *out = a * b;
        return true;
    };
    auto add = [](size_t* out, size_t value) {
        if (value > std::numeric_limits<size_t>::max() - *out) return false;
        *out += value;
        return true;
    };
    size_t curveOutput = 0;
    if (!multiply(curves, rootBytes, &candidate.inputBytes) ||
        !multiply(candidate.pointCount, pointBytes, &candidate.outputBytes) ||
        !multiply(curves, curveBytes, &curveOutput) ||
        !add(&candidate.outputBytes, curveOutput) ||
        !add(&candidate.outputBytes, sizeof(uint32_t)))
        return ScatterGrowStatus::InvalidTopology;
    candidate.peakBytes = candidate.inputBytes;
    if (!add(&candidate.peakBytes, candidate.outputBytes))
        return ScatterGrowStatus::InvalidTopology;
    candidate.statusBytes = 2 * sizeof(int); // device status + pinned relay status
    if (!add(&candidate.peakBytes, candidate.statusBytes))
        return ScatterGrowStatus::InvalidTopology;
    *result = candidate;
    return ScatterGrowStatus::Ok;
}

void CudaScatterGrow::Storage::quarantine() noexcept { points.quarantine(); rootT.quarantine(); rootB.quarantine(); rootN.quarantine(); widths.quarantine(); hairT.quarantine(); offsets.quarantine(); stableIds.quarantine(); rootPrim.quarantine(); rootUV.quarantine(); }
size_t CudaScatterGrow::Storage::bytes() const noexcept { return points.bytes()+rootT.bytes()+rootB.bytes()+rootN.bytes()+widths.bytes()+hairT.bytes()+offsets.bytes()+stableIds.bytes()+rootPrim.bytes()+rootUV.bytes(); }
void CudaScatterGrow::Storage::Reclassify(UsdGenExecutionResourceKind kind) noexcept { points.Reclassify(kind); rootT.Reclassify(kind); rootB.Reclassify(kind); rootN.Reclassify(kind); widths.Reclassify(kind); hairT.Reclassify(kind); offsets.Reclassify(kind); stableIds.Reclassify(kind); rootPrim.Reclassify(kind); rootUV.Reclassify(kind); }
ScatterGrowStatus CudaScatterGrow::Storage::recordUse(cudaStream_t s) { cudaError_t e=points.recordUse(s); if(e==cudaSuccess)e=widths.recordUse(s); if(e==cudaSuccess)e=hairT.recordUse(s); if(e==cudaSuccess)e=offsets.recordUse(s); if(e==cudaSuccess)e=stableIds.recordUse(s); if(e==cudaSuccess)e=rootPrim.recordUse(s); if(e==cudaSuccess)e=rootUV.recordUse(s); if(e==cudaSuccess)e=rootT.recordUse(s); if(e==cudaSuccess)e=rootB.recordUse(s); if(e==cudaSuccess)e=rootN.recordUse(s); return Status(e); }
ScatterGrowStatus CudaScatterGrow::Storage::waitOn(cudaStream_t s) const { cudaError_t e=points.waitOn(s); if(e==cudaSuccess)e=widths.waitOn(s); if(e==cudaSuccess)e=hairT.waitOn(s); if(e==cudaSuccess)e=offsets.waitOn(s); if(e==cudaSuccess)e=stableIds.waitOn(s); if(e==cudaSuccess)e=rootPrim.waitOn(s); if(e==cudaSuccess)e=rootUV.waitOn(s); if(e==cudaSuccess)e=rootT.waitOn(s); if(e==cudaSuccess)e=rootB.waitOn(s); if(e==cudaSuccess)e=rootN.waitOn(s); return Status(e); }
ScatterGrowStatus CudaScatterGrow::Storage::synchronizeUse() const { cudaError_t e=points.synchronizeUse(); if(e==cudaSuccess)e=widths.synchronizeUse(); if(e==cudaSuccess)e=hairT.synchronizeUse(); if(e==cudaSuccess)e=offsets.synchronizeUse(); if(e==cudaSuccess)e=stableIds.synchronizeUse(); if(e==cudaSuccess)e=rootPrim.synchronizeUse(); if(e==cudaSuccess)e=rootUV.synchronizeUse(); if(e==cudaSuccess)e=rootT.synchronizeUse(); if(e==cudaSuccess)e=rootB.synchronizeUse(); if(e==cudaSuccess)e=rootN.synchronizeUse(); return Status(e); }

CudaScatterGrow::~CudaScatterGrow() {
    if (unprovenWork_) {
        active_.quarantine(); pending_.quarantine(); pendingInput_.quarantine(); error_.quarantine();
        // The device destination and immutable host source belong to the
        // same unproved transfer. Neither may be freed on this path.
        (void)rootsQuarantineOwner_.release();
        hostError_ = nullptr;
        hostErrorPermit_.Abandon();
        ready_ = nullptr; return;
    }
    int prior=-1; bool owns=active_.points.size()||pending_.points.size()||error_.size()||
        ready_||hostError_;
    bool selected=!owns || (cudaGetDevice(&prior)==cudaSuccess && deviceIndex_>=0 && cudaSetDevice(deviceIndex_)==cudaSuccess);
    bool proved=!owns || (selected && (!ready_ || cudaEventSynchronize(ready_)==cudaSuccess) &&
        active_.synchronizeUse()==ScatterGrowStatus::Ok && error_.synchronizeUse()==cudaSuccess);
    if (!proved) {
        active_.quarantine(); pending_.quarantine(); pendingInput_.quarantine(); error_.quarantine();
        (void)rootsQuarantineOwner_.release();
        hostError_ = nullptr;
        hostErrorPermit_.Abandon();
        ready_=nullptr;
    } else {
        if (ready_) cudaEventDestroy(ready_);
        if (hostError_) {
            // Proven complete (see above): the relay is quiescent, so it
            // rejoins the pool and the permit releases as on a free.
            hostErrorPermit_.Release();
            PushPinnedRelay(hostError_);
            hostError_ = nullptr;
        }
    }
    if (selected && prior>=0 && prior!=deviceIndex_) cudaSetDevice(prior);
}

ScatterGrowStatus CudaScatterGrow::validateStream(cudaStream_t stream) const {
    // Preflight belongs before any allocation: a first-use foreign/captured
    // stream must not bind this owner or enqueue root copies.
    int current=-1; if(cudaGetDevice(&current)!=cudaSuccess)return ScatterGrowStatus::CudaError;
    if(deviceIndex_>=0 && current!=deviceIndex_)return ScatterGrowStatus::InvalidArgument;
    cudaStreamCaptureStatus capture;
    if(cudaStreamIsCapturing(stream,&capture)!=cudaSuccess)return ScatterGrowStatus::CudaError;
    if(capture!=cudaStreamCaptureStatusNone)return ScatterGrowStatus::InvalidArgument;
    // cudaStreamGetDevice is not capture-safe on every supported runtime.
    if(stream) { int d=-1; if(cudaStreamGetDevice(stream,&d)!=cudaSuccess)return ScatterGrowStatus::CudaError; if(d!=current || (deviceIndex_>=0 && d!=deviceIndex_))return ScatterGrowStatus::InvalidArgument; }
    return ScatterGrowStatus::Ok;
}
ScatterGrowStatus CudaScatterGrow::validate(
    std::shared_ptr<const ScatterGrowRoots> const& r, ScatterGrowControls const& c,
    size_t* total) const {
    return ValidateRoots(r, c, total);
}
ScatterGrowStatus CudaScatterGrow::ValidateRoots(
    std::shared_ptr<const ScatterGrowRoots> const& r, ScatterGrowControls const& c,
    size_t* total) {
    size_t n = 0, checkedTotal = 0;
    ScatterGrowStatus const pre = PrecheckRoots(r, c, &n, &checkedTotal);
    if (pre != ScatterGrowStatus::Ok)
        return pre;
    *total = checkedTotal;
    // Minima-combine validator: the fused loop returned the first fault
    // in index order (finite-inputs, then duplicate, then overflow math
    // within an index), which is the minimum of three first-fault
    // indices with that tie-break. Each leg is cache-streaming; the old
    // fused loop probed a 36MB dup set at random under a 56MB stream.
    // Finite-inputs leg: first failing index, verbatim condition.
    // Multi-pass scan (verdict-identical): each plane's first non-finite
    // float independently bounds the first failing root (a bad lane in
    // root r's floats makes r bad; all-clean floats before it make every
    // earlier root clean), so fBad is the minimum over planes. The inner
    // exponent-OR scans 16 floats per iteration under one branch; a
    // firing OR re-scans its group scalar for the exact first lane (an
    // OR combination can false-positive but never false-negative, since
    // any 0xFF lane sets the OR's exponent field).
    auto firstBadRoot = [](uint32_t const *u, size_t roots, size_t stride,
                           size_t lim) -> size_t {
        size_t const words = std::min(roots, lim) * stride;
        size_t base = 0;
        size_t const w16 = words & ~size_t(15);
        for (; base < w16; base += 16) {
#if defined(__aarch64__) && defined(__GNUC__)
            // Vector OR over the 16-word group (verdict-identical): the
            // same 16 words feed one OR, then the same exponent test and
            // the same scalar rescan below. Replaces 16 scalar loads + a
            // serial 8-deep OR chain with 4 vector loads + a 2-level OR
            // tree; the lane extraction is plain ORs in another order.
            // GCC vector extensions (not arm_neon.h: nvcc's frontend
            // cannot parse that header); memcpy loads are alignment-safe.
            typedef uint32_t u32x4 __attribute__((vector_size(16)));
            u32x4 v0, v1, v2, v3;
            std::memcpy(&v0, u + base + 0, sizeof(v0));
            std::memcpy(&v1, u + base + 4, sizeof(v1));
            std::memcpy(&v2, u + base + 8, sizeof(v2));
            std::memcpy(&v3, u + base + 12, sizeof(v3));
            u32x4 const o = v0 | v1 | v2 | v3;
            uint32_t const acc = o[0] | o[1] | o[2] | o[3];
#else
            uint32_t acc = 0;
            for (size_t k = 0; k < 16; ++k) acc |= u[base + k];
#endif
            if ((acc & 0x7F800000u) == 0x7F800000u) {
                for (size_t k = 0; k < 16; ++k) {
                    if ((u[base + k] & 0x7F800000u) == 0x7F800000u)
                        return (base + k) / stride;
                }
            }
        }
        for (; base < words; ++base) {
            if ((u[base] & 0x7F800000u) == 0x7F800000u)
                return base / stride;
        }
        return lim;
    };
    // Fused positions scan (verdict-identical): the bounded-math fast
    // path below needs max |position| over a superset of [0, mLim), and
    // this scan already streams every positions word, so it accumulates
    // that max over the same loads instead of re-streaming positions.
    // Integer max over sign-cleared bits equals the old fmax chain on
    // finite lanes (magnitude order is integer order, subnormals and
    // +-0 included), so clean inputs take the identical branch. The
    // scanned prefix always covers [0, mLim): mLim <= fBad <= the
    // first-bad root, and a clean scan covers all n. A non-finite lane
    // in the prefix (exponent 0xFF, the integer maximum) fails the
    // bound check toward the slow path, which is the verdict reference.
    auto firstBadRootMax = [](uint32_t const *u, size_t roots, size_t stride,
                              size_t lim, uint32_t *maxBits) -> size_t {
        size_t const words = std::min(roots, lim) * stride;
        size_t base = 0;
        size_t const w16 = words & ~size_t(15);
#if defined(__aarch64__) && defined(__GNUC__)
        typedef uint32_t u32x4 __attribute__((vector_size(16)));
        u32x4 macc = {0u, 0u, 0u, 0u};
#endif
        uint32_t mx = 0;
        for (; base < w16; base += 16) {
#if defined(__aarch64__) && defined(__GNUC__)
            u32x4 v0, v1, v2, v3;
            std::memcpy(&v0, u + base + 0, sizeof(v0));
            std::memcpy(&v1, u + base + 4, sizeof(v1));
            std::memcpy(&v2, u + base + 8, sizeof(v2));
            std::memcpy(&v3, u + base + 12, sizeof(v3));
            u32x4 const o = v0 | v1 | v2 | v3;
            uint32_t const acc = o[0] | o[1] | o[2] | o[3];
            u32x4 const m0 = v0 & 0x7FFFFFFFu;
            u32x4 const m1 = v1 & 0x7FFFFFFFu;
            u32x4 const m2 = v2 & 0x7FFFFFFFu;
            u32x4 const m3 = v3 & 0x7FFFFFFFu;
            u32x4 const hi01 = m0 > m1 ? m0 : m1;
            u32x4 const hi23 = m2 > m3 ? m2 : m3;
            u32x4 const hi = hi01 > hi23 ? hi01 : hi23;
            macc = macc > hi ? macc : hi;
#else
            uint32_t acc = 0;
            for (size_t k = 0; k < 16; ++k) {
                uint32_t const w = u[base + k];
                acc |= w;
                uint32_t const mag = w & 0x7FFFFFFFu;
                mx = mag > mx ? mag : mx;
            }
#endif
            if ((acc & 0x7F800000u) == 0x7F800000u) {
#if defined(__aarch64__) && defined(__GNUC__)
                uint32_t lanes[4];
                std::memcpy(lanes, &macc, sizeof(lanes));
                for (int q = 0; q < 4; ++q)
                    mx = lanes[q] > mx ? lanes[q] : mx;
#endif
                for (size_t k = 0; k < 16; ++k) {
                    uint32_t const w = u[base + k];
                    uint32_t const mag = w & 0x7FFFFFFFu;
                    mx = mag > mx ? mag : mx;
                    if ((w & 0x7F800000u) == 0x7F800000u) {
                        *maxBits = mx;
                        return (base + k) / stride;
                    }
                }
            }
        }
        for (; base < words; ++base) {
            uint32_t const w = u[base];
            uint32_t const mag = w & 0x7FFFFFFFu;
            mx = mag > mx ? mag : mx;
            if ((w & 0x7F800000u) == 0x7F800000u) {
                *maxBits = mx;
                return base / stride;
            }
        }
#if defined(__aarch64__) && defined(__GNUC__)
        uint32_t lanes[4];
        std::memcpy(lanes, &macc, sizeof(lanes));
        for (int q = 0; q < 4; ++q) mx = lanes[q] > mx ? lanes[q] : mx;
#endif
        *maxBits = mx;
        return lim;
    };
    // Threaded finite-input scans for big inputs (same n>32768 rule
    // as the dup leg; small inputs keep the serial spelling): each
    // plane's scan is an independent function of its own plane, and
    // the serial lim-threading only bounds work -- firstBad(u, lim)
    // is min(firstBad(u, n), lim) -- so scanning every plane at lim=n
    // and min-combining is verdict-identical. The positions scan is
    // byte-for-byte the serial call (same lim=n it always ran with),
    // so posMaxBits is identical too. Plain TBB.
    size_t fBad = n;
    uint32_t posMaxBits = 0;
    uint32_t const *posW =
        reinterpret_cast<uint32_t const *>(r->positions.data());
    uint32_t const *uvW =
        reinterpret_cast<uint32_t const *>(r->rootUV.data());
    uint32_t const *tW =
        reinterpret_cast<uint32_t const *>(r->rootT.data());
    uint32_t const *bW =
        reinterpret_cast<uint32_t const *>(r->rootB.data());
    uint32_t const *nW =
        reinterpret_cast<uint32_t const *>(r->rootN.data());
    int const scanWorkers = tbb::this_task_arena::max_concurrency();
    if (scanWorkers > 1 && n > 32768) {
        size_t planeBad[5] = {n, n, n, n, n};
        uint32_t planeMax = 0;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, 5),
            [&](tbb::blocked_range<size_t> const &range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    switch (i) {
                    case 0: {
                        uint32_t mx = 0;
                        planeBad[0] =
                            firstBadRootMax(posW, n, 3, n, &mx);
                        planeMax = mx;
                        break;
                    }
                    case 1:
                        planeBad[1] = firstBadRoot(uvW, n, 2, n);
                        break;
                    case 2:
                        planeBad[2] = firstBadRoot(tW, n, 3, n);
                        break;
                    case 3:
                        planeBad[3] = firstBadRoot(bW, n, 3, n);
                        break;
                    default:
                        planeBad[4] = firstBadRoot(nW, n, 3, n);
                        break;
                    }
                }
            });
        posMaxBits = planeMax;
        for (int i = 0; i < 5; ++i)
            fBad = planeBad[i] < fBad ? planeBad[i] : fBad;
    } else {
        fBad = firstBadRootMax(posW, n, 3, fBad, &posMaxBits);
        fBad = firstBadRoot(uvW, n, 2, fBad);
        fBad = firstBadRoot(tW, n, 3, fBad);
        fBad = firstBadRoot(bW, n, 3, fBad);
        fBad = firstBadRoot(nW, n, 3, fBad);
    }
    // Duplicate leg: minimum second-occurrence index (radix; the set
    // form only past 32-bit index range).
    size_t const dBad = n > uint64_t(std::numeric_limits<uint32_t>::max())
        ? SetFirstDup(r->stableIds.data(), n)
        : RadixFirstDup(r->stableIds.data(), n);
    // Overflow-math leg, verbatim per-index ops over [0, min(fBad,dBad)):
    // past that bound an earlier-or-tied fault of higher-or-equal
    // precedence already wins (finite ties beat dup ties beat math
    // ties), so later math faults are moot.
    size_t const mLim = std::min(fBad, dBad);
    size_t mBad = mLim;
    // Bounded-math fast path (verdict-identical): the overflow leg below
    // can only fault through target or output overflow, and both are
    // provably finite when the controls and the position range are
    // small. NormalizeHost maps any finite input to components <=
    // 1.00001 (its length is the input's own, so each quotient is ~1;
    // tiny/NaN/huge lengths take the finite fallback branches), and each
    // RotateAroundBHost of such vectors stays <= 9x its input (bounded
    // dot/cross/cos/sin terms), so the grown direction is <= 82 for any
    // validated lift/azimuth. The target is length * a [lo,hi] lerp of a
    // [0,1) draw, finite when tBound = length*max(lo,hi) <= FLT_MAX/8;
    // the output is pos + dir*distance with |dir| <= 128 (margin over
    // 82), finite when Pmax + 128*tBound <= FLT_MAX/4. Rounding moves
    // every bound by ~1e-7 relative against 4-8x margins. NaN positions
    // are fmax-invisible yet still break the finite leg first (mLim <=
    // fBad), so the skipped domain stays all-finite; an Inf Pmax fails
    // the comparison and takes the slow path. A taken fast path means
    // mBad == mLim exactly as a clean slow run finds.
    double const tBound = c.length * std::max(c.randomLo, c.randomHi);
    constexpr double kTargetLim =
        double(std::numeric_limits<float>::max()) / 8.0;
    size_t mRun = mLim;
    if (tBound <= kTargetLim) {
        // Max |position| comes from the fused positions scan above (a
        // superset of [0, mLim), so a conservative bound): integer max
        // over sign-cleared bits, equal to the old fmax chain on
        // finite lanes. A non-finite lane in the prefix poisons the
        // bits toward Inf/NaN, which fails the check below toward the
        // slow path instead of taking the fast path.
        float posMaxF = 0.0f;
        std::memcpy(&posMaxF, &posMaxBits, sizeof(float));
        double const posMax = double(posMaxF);
        constexpr double kOutLim =
            double(std::numeric_limits<float>::max()) / 4.0;
        if (posMax + 128.0 * tBound <= kOutLim) mRun = 0;
    }
    // Hoisted draw seed folds (bit-identical integer CSE): c.seed is
    // loop-invariant, so each salt's Hash64Host(seed32, salt) runs once.
    uint64_t const seed32 = uint64_t(uint32_t(c.seed));
    uint64_t const hSeedAz = Hash64Host(seed32, 0x4772417Au);
    uint64_t const hSeed = Hash64Host(seed32, 0x47726F77u);
    // The azimuth draw is dead when azimuthRandom is +-0 (bit-identical):
    // the computed azimuth is c.azimuth plus a +-0 product, which equals
    // c.azimuth except for a -0/+0 edge that RotateAroundBHost's
    // degrees==0 early-out treats identically (c.azimuth is validated
    // finite, so no NaN/Inf lane can differ).
    bool const noAzimuthRandom = c.azimuthRandom == 0.0f;
    for(size_t i=0;i<mRun;++i) {
        // Catch deterministic target and output overflow before reserving or
        // submitting any work.  The device repeats this check and reports a
        // native status as well, since float contraction can differ at the
        // last bit near FLT_MAX.
        float3 direction = c.direction == ScatterGrowDirection::RootNormal ? r->rootN[i] :
            c.direction == ScatterGrowDirection::RootTangent ? r->rootT[i] : c.literalDirection;
        direction = RotateAroundBHost(NormalizeHost(direction), r->rootB[i], c.lift);
        float const azimuth = noAzimuthRandom ? c.azimuth :
            c.azimuth + c.azimuthRandom * 360.0f *
            (DrawGrowHostSeeded(hSeedAz, r->stableIds[i], 0x4772417Au) - 0.5f);
        direction = RotateAroundBHost(direction, r->rootN[i], azimuth);
        if (!Finite(direction)) { mBad = i; break; }
        double const targetDouble = c.length *
            (c.randomLo + double(DrawGrowHostSeeded(hSeed, r->stableIds[i],
                                                    0x47726F77u)) *
             (c.randomHi - c.randomLo));
        float const target = static_cast<float>(targetDouble);
        if (!Finite(target)) { mBad = i; break; }
        // Only the last iteration (t = 1) can report NonFiniteInput, so
        // the loop over j runs once, spelled verbatim: distance_j =
        // target * t_j with t_j in [0,1] can neither overflow (its
        // magnitude is at most |target|, finite) nor go non-finite, and
        // the output sum pos + dir*distance_j overflows at some j only
        // if it overflows at full extension (same-sign sums grow with
        // |distance|; opposite-sign sums stay within the larger of the
        // two finite magnitudes). The device repeats the full loop
        // unchanged.
        {
            // t is exactly 1 (cvCount-1 is an exact small float, so x/x
            // is 1) and target*1 is target, so distance is target and
            // its finiteness is the target check above, verbatim values.
            float const distance = target;
            float3 const output = make_float3(
                r->positions[i].x + direction.x * distance,
                r->positions[i].y + direction.y * distance,
                r->positions[i].z + direction.z * distance);
            if (!Finite(output))
                { mBad = i; break; }
        }
    }
    // Precedence combine: a math fault strictly inside the bound is the
    // global first fault; otherwise the finite/dup minima decide with
    // finite-inputs winning ties, exactly the fused loop's order.
    if (mBad < mLim) return ScatterGrowStatus::NonFiniteInput;
    if (fBad <= dBad)
        return fBad < n ? ScatterGrowStatus::NonFiniteInput
                        : ScatterGrowStatus::Ok;
    return ScatterGrowStatus::DuplicateStableId;
}
void CudaScatterGrow::discardPending() noexcept {
    pending_=Storage{};
    pendingInput_=Storage{};
    error_.reset(0);
    if (hostError_) {
        // Pre-issue failure, or the post-proof error commit: no work is in
        // flight on the relay, so it rejoins the pool and the permit
        // releases as on a free.
        hostErrorPermit_.Release();
        PushPinnedRelay(hostError_);
        hostError_ = nullptr;
    }
    rootsOwner_.reset(); rootsQuarantineOwner_.reset();
    pendingWork_=finishScheduled_=false; pendingCurves_=pendingPoints_=0;
}
ScatterGrowStatus CudaScatterGrow::BeginFresh(
    std::shared_ptr<const ScatterGrowRoots> roots, ScatterGrowControls controls,
    cudaStream_t stream, UsdGenExecutionMemoryReservation* reserve) {
    if(pendingWork_||generation_)return ScatterGrowStatus::InvalidArgument; auto s=validateStream(stream); if(s!=ScatterGrowStatus::Ok)return s; if(controls.randomLo>controls.randomHi)std::swap(controls.randomLo,controls.randomHi);
    // Big grows overlap the H2D copies (pageable sources: ~1.2ms of
    // synchronous CPU-side staging per 1M roots) with the validation
    // scans (~1.8ms) on a worker thread: spec-allocate from the cheap
    // validation prefix, then copies and scans run side by side. Same
    // bytes, same stream order, same statuses. Small grows (and null
    // roots) keep today's exact serial order: validate first, so failure
    // has no side effects and no spawn/join dwarfs small copies.
    size_t const peekN = roots ? roots->positions.size() : 0;
    size_t total = 0;
    // The small path validates up front (today's order); the big path
    // defers the scans to the overlap below. The serial branch below
    // re-checks, so it stays correct if the overlap is ever disabled.
    bool const validated = peekN <= 65536;
    if (validated) {
        s = ValidateRoots(roots, controls, &total);
        if (s != ScatterGrowStatus::Ok)
            return s;
    } else {
        size_t n = 0;
        s = PrecheckRoots(roots, controls, &n, &total);
        if (s != ScatterGrowStatus::Ok)
            return ValidateRoots(roots, controls, &total);
    }
    bool const overlapCopies = peekN > 65536;
    int d=-1; if(cudaGetDevice(&d)!=cudaSuccess)return ScatterGrowStatus::CudaError; if(deviceIndex_<0)deviceIndex_=d;
    Storage in; // temporary input storage, kept alive through terminal proof.
    // Positions alone stage here: ids/prim/uv/frames upload directly into
    // their published pending_ buffers below, so the kernel's per-thread
    // copy tail (and its second allocation of the same bytes) is gone.
    cudaError_t e=Allocate(in.points,roots->positions.size(),reserve);
    if(e!=cudaSuccess)return Status(e);
    e=pending_.points.reset(total,reserve); if(e==cudaSuccess)e=pending_.widths.reset(total,reserve); if(e==cudaSuccess)e=pending_.hairT.reset(total,reserve); if(e==cudaSuccess)e=pending_.offsets.reset(roots->positions.size()+1,reserve); if(e==cudaSuccess)e=pending_.stableIds.reset(roots->positions.size(),reserve); if(e==cudaSuccess)e=pending_.rootPrim.reset(roots->positions.size(),reserve); if(e==cudaSuccess)e=pending_.rootUV.reset(roots->positions.size(),reserve); if(e==cudaSuccess)e=pending_.rootT.reset(roots->positions.size(),reserve); if(e==cudaSuccess)e=pending_.rootB.reset(roots->positions.size(),reserve); if(e==cudaSuccess)e=pending_.rootN.reset(roots->positions.size(),reserve); if(e==cudaSuccess)e=error_.reset(1,reserve,UsdGenExecutionResourceKind::Scratch); if(e!=cudaSuccess){discardPending(); return Status(e);}
    if (!hostError_) {
        auto permit = TryReserveCudaExecutionBytes(sizeof(int),
            UsdGenExecutionResourceKind::Scratch, reserve);
        int* relay = PopPinnedRelay();
        if (!permit || (!relay && cudaHostAlloc(reinterpret_cast<void**>(&relay), sizeof(int),
                                                cudaHostAllocDefault) != cudaSuccess)) {
            if (relay) PushPinnedRelay(relay);
            discardPending();
            return ScatterGrowStatus::CudaError;
        }
        hostError_ = relay;
        hostErrorPermit_ = std::move(*permit);
    }
    try {
        rootsQuarantineOwner_ = std::make_unique<std::shared_ptr<const ScatterGrowRoots>>(roots);
    } catch (...) {
        discardPending();
        return ScatterGrowStatus::CudaError;
    }
    rootsOwner_=std::move(roots); pendingCurves_=rootsOwner_->positions.size(); pendingPoints_=total; pendingWork_=true;
    producerStream_ = stream;
    // From this point H2D copies and the kernel may be in flight.  A later
    // failure has no completion proof and must quarantine rather than free.
    pendingInput_=std::move(in); unprovenWork_=true;
    *hostError_ = kPendingStatus;
    // One spelling for the memset + seven H2D copies, issued serially on
    // the small path or on the overlap worker below. Stream order is
    // unchanged either way (memset, copies, kernel).
    auto issueCopies = [&]() -> cudaError_t {
        cudaError_t ce = cudaMemsetAsync(error_.data(), 0, sizeof(int), stream);
        if (ce == cudaSuccess) ce=Copy(pendingInput_.points,rootsOwner_->positions.data(),rootsOwner_->positions.size(),stream); if(ce==cudaSuccess)ce=Copy(pending_.stableIds,rootsOwner_->stableIds.data(),rootsOwner_->stableIds.size(),stream); if(ce==cudaSuccess)ce=Copy(pending_.rootPrim,rootsOwner_->rootPrim.data(),rootsOwner_->rootPrim.size(),stream); if(ce==cudaSuccess)ce=Copy(pending_.rootUV,rootsOwner_->rootUV.data(),rootsOwner_->rootUV.size(),stream); if(ce==cudaSuccess)ce=Copy(pending_.rootT,rootsOwner_->rootT.data(),rootsOwner_->rootT.size(),stream); if(ce==cudaSuccess)ce=Copy(pending_.rootB,rootsOwner_->rootB.data(),rootsOwner_->rootB.size(),stream); if(ce==cudaSuccess)ce=Copy(pending_.rootN,rootsOwner_->rootN.data(),rootsOwner_->rootN.size(),stream);
        return ce;
    };
    // One spelling for the grow launch: the overlap worker fires it right
    // after the copies while the validation scans still run (below); the
    // small path fires it after the scans as today. Same stream order,
    // same arguments, same error reporting either way.
    auto launchGrow = [&]() -> cudaError_t {
        GrowKernel<<<(unsigned(pendingCurves_)+127)/128,128,0,stream>>>(pendingInput_.points.data(),pending_.stableIds.data(),pending_.rootT.data(),pending_.rootB.data(),pending_.rootN.data(),uint32_t(pendingCurves_),controls.cvCount,controls.seed,controls.length,controls.randomLo,controls.randomHi,controls.lift,controls.azimuth,controls.azimuthRandom,controls.fallbackWidth,controls.direction,controls.literalDirection,pending_.points.data(),pending_.widths.data(),pending_.hairT.data(),pending_.offsets.data(),error_.data());
        return cudaGetLastError();
    };
    if (overlapCopies) {
        // The worker issues the synchronous staging copies and fires the
        // grow kernel while this thread runs the validation scans; the
        // join proves both done. The kernel launch is speculative but
        // sound: stream order (memset, copies, kernel) and kernel
        // arguments match today's serial spelling exactly, so the
        // success path is bit-identical and the device-side error flag
        // reads the same in CommitFreshFinish. A failed validation
        // verdict still returns before anything is published: the sync
        // below proves the whole stream (copies and kernel) idle and
        // the object discards clean and reusable, exactly like today's
        // pre-alloc validation failure. The API stays synchronous:
        // BeginFresh returns only after the validation verdict. Both
        // threads read rootsOwner_ without mutating it, and the
        // buffers are untouched on this thread until the join, so no
        // lock is needed. Device errors keep today's unproven
        // discipline (a double fault reports the copy error, then the
        // launch error, not the scan verdict). The overlap path always
        // has curves (peekN > 65536), so the worker always launches.
        cudaError_t copyError = cudaSuccess;
        cudaError_t launchError = cudaSuccess;
        std::thread copyThread;
        try {
            copyThread = std::thread(
                [this, &issueCopies, &launchGrow, &copyError, &launchError]() {
                cudaError_t ce = cudaSetDevice(deviceIndex_);
                if (ce == cudaSuccess)
                    ce = issueCopies();
                copyError = ce;
                if (ce == cudaSuccess)
                    launchError = launchGrow();
            });
        } catch (...) {
            // Nothing issued yet: discard like the pre-issue failures.
            discardPending();
            return ScatterGrowStatus::CudaError;
        }
        try {
            s = validate(rootsOwner_, controls, &total);
        } catch (...) {
            // Join first (a joinable thread must never unwind past),
            // then leave the object as clean as today's pre-alloc
            // validation throw: the join proves the synchronous copies
            // complete and the sync proves the memset and the kernel,
            // so discarding is safe. The throw itself propagates
            // unchanged.
            copyThread.join();
            if (cudaStreamSynchronize(stream) == cudaSuccess) {
                unprovenWork_ = false;
                discardPending();
            }
            throw;
        }
        copyThread.join();
        if (copyError != cudaSuccess) return Status(copyError);
        if (launchError != cudaSuccess) return Status(launchError);
        if (s != ScatterGrowStatus::Ok) {
            // The join proves the synchronous copies complete, but the
            // memset and the kernel may still be in flight: prove the
            // stream idle before discarding, so the object stays clean
            // and reusable exactly like today's pre-alloc validation
            // failure. Failure path only.
            cudaError_t const syncE = cudaStreamSynchronize(stream);
            if (syncE != cudaSuccess) return Status(syncE);
            unprovenWork_ = false;
            discardPending();
            return s;
        }
    } else {
        if (!validated) {
            // Big grow with the overlap disabled: run the scans first,
            // so the serial branch always validates before copying.
            // Nothing is issued yet, so a failure discards cleanly.
            s = validate(rootsOwner_, controls, &total);
            if (s != ScatterGrowStatus::Ok) {
                discardPending();
                return s;
            }
        }
        e = issueCopies();
        if (e != cudaSuccess) return Status(e);
    }
    if (!pendingCurves_) {
        e = cudaMemsetAsync(pending_.offsets.data(), 0, sizeof(uint32_t), stream);
        if (e != cudaSuccess) return Status(e);
        return ScatterGrowStatus::Ok;
    }
    if (!overlapCopies) {
        // Small path: launch after the scans as today (the overlap
        // worker already launched above).
        e = launchGrow();
        if (e != cudaSuccess) return Status(e);
    }
    // rest == points elementwise (the kernel writes every point), and the
    // published generation is immutable, so view() aliases the points
    // buffer under the rest slot: no D2D rest copy, no second allocation.
    // Identical bytes on the success path; the error path discards points.
    return ScatterGrowStatus::Ok;
}

ScatterGrowStatus CudaScatterGrow::FinishFreshAsync(
    cudaStream_t stream,
    void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata) {
    if (!pendingWork_ || finishScheduled_ || !callback)
        return ScatterGrowStatus::NoPendingUpdate;
    if (stream != producerStream_) return ScatterGrowStatus::InvalidArgument;
    auto status = validateStream(stream);
    if (status != ScatterGrowStatus::Ok) return status;
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return ScatterGrowStatus::CudaError;
    status = pending_.recordUse(stream);
    if (status == ScatterGrowStatus::Ok) status = pendingInput_.recordUse(stream);
    if (status == ScatterGrowStatus::Ok) status = Status(error_.recordUse(stream));
    if (status != ScatterGrowStatus::Ok) return status;
    auto error = cudaEventRecord(ready_, stream);
    if (error != cudaSuccess) return Status(error);
    error = cudaMemcpyAsync(hostError_, error_.data(), sizeof(int),
                            cudaMemcpyDeviceToHost, stream);
    if (error != cudaSuccess) return Status(error);
    error = cudaStreamAddCallback(stream, callback, userdata, 0);
    // A rejected installation does not authorize commit. The caller still
    // owns the unproved producer and must retain it or establish cleanup proof.
    if (error == cudaSuccess) finishScheduled_ = true;
    return Status(error);
}
ScatterGrowStatus CudaScatterGrow::CommitFreshFinish() {
    if(!pendingWork_||!finishScheduled_) return ScatterGrowStatus::NoPendingUpdate;
    if (!hostError_ || *hostError_ == kPendingStatus)
        return ScatterGrowStatus::NoPendingUpdate;
    if (*hostError_ != 0) {
        ScatterGrowStatus const status = *hostError_ == kNonFinite
            ? ScatterGrowStatus::NonFiniteInput : ScatterGrowStatus::InvalidTopology;
        unprovenWork_ = false;
        discardPending();
        return status;
    }
    active_=std::move(pending_); pendingInput_=Storage{};
    rootsOwner_.reset(); rootsQuarantineOwner_.reset();
    curves_=pendingCurves_; points_=pendingPoints_;
    pendingCurves_=pendingPoints_=0; pendingWork_=finishScheduled_=false;
    unprovenWork_=false; ++generation_; return ScatterGrowStatus::Ok;
}
DeviceCurveGeometryView CudaScatterGrow::view() const { auto const pts=active_.points.view(); return {pts,pts,active_.widths.view(),active_.offsets.view(),active_.stableIds.view(),curves_,points_}; }
DeviceView<const float> CudaScatterGrow::hairT() const{return active_.hairT.view();} DeviceView<const int32_t> CudaScatterGrow::rootPrim() const{return active_.rootPrim.view();} DeviceView<const float2> CudaScatterGrow::rootUV() const{return active_.rootUV.view();} DeviceView<const float3> CudaScatterGrow::rootT() const{return active_.rootT.view();} DeviceView<const float3> CudaScatterGrow::rootB() const{return active_.rootB.view();} DeviceView<const float3> CudaScatterGrow::rootN() const{return active_.rootN.view();}
ScatterGrowStatus CudaScatterGrow::recordUse(cudaStream_t s){auto x=validateStream(s);return x==ScatterGrowStatus::Ok?active_.recordUse(s):x;} ScatterGrowStatus CudaScatterGrow::waitOn(cudaStream_t s) const{auto x=validateStream(s);return x==ScatterGrowStatus::Ok?active_.waitOn(s):x;} size_t CudaScatterGrow::ExclusiveRetainedBytes() const noexcept{return active_.bytes()+error_.bytes()+hostErrorPermit_.Bytes();}
void CudaScatterGrow::ReclassifyPublishedGeneration() noexcept {
    active_.Reclassify(UsdGenExecutionResourceKind::Pinned);
    error_.Reclassify(UsdGenExecutionResourceKind::Pinned);
    hostErrorPermit_.Reclassify(UsdGenExecutionResourceKind::Pinned);
}
} // namespace usdGen::gpu
