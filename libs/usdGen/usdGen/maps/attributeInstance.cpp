// usdGen — authored attribute maps into groom cooking/instancing.
// See attributeInstance.h for the contract and its distance from Pomade.
#include "usdGen/maps/attributeInstance.h"

#include "usdGen/digest.h"

#include "tbb/parallel_for.h"
#include "tbb/task_arena.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace usdGen {

bool UsdGenAttributeCookInstances(UsdGenAttributeInstanceInput const &input,
                                  UsdGenAttributeInstanceResult *result,
                                  std::string *error)
{
    auto fail = [&](std::string const &message) {
        if (error) *error = message;
        return false;
    };
    if (!input.map) return fail("instance cook needs a map");
    if (input.channel < 0 || input.channel >= input.map->Channels())
        return fail("instance cook channel is outside the map");
    if (input.roots.empty()) return fail("instance cook needs at least one root");
    if (input.numPrototypes < 1)
        return fail("instance cook needs numPrototypes >= 1");
    if (!std::isfinite(input.threshold))
        return fail("instance cook threshold must be finite");
    if (!std::isfinite(input.defaultValue))
        return fail("instance cook default must be finite");
    if (!result) return fail("instance cook needs a result");

    UsdGenAttributeInstanceResult cooked;
    size_t const count = input.roots.size();
    cooked.values.resize(count);
    cooked.keep.resize(count);
    cooked.prototype.resize(count);
    cooked.instanceIndices.resize(size_t(input.numPrototypes));
    // Pre-size the per-slot index lists (bit-identical: same pushed values,
    // no reallocation copies): every kept root lands in exactly one slot,
    // so count slots total suffice however skewed the distribution is. The
    // uniform split covers the balanced case with zero growth; a skewed
    // slot resumes geometric growth past its share, never worse than the
    // unreserved baseline. Bounded by the kept total, not count times the
    // slot count, so a huge prototype count cannot over-reserve.
    size_t const perSlot =
        (count + size_t(input.numPrototypes) - 1) / size_t(input.numPrototypes);
    for (std::vector<int> &slot : cooked.instanceIndices)
        slot.reserve(perSlot);

    // Inline UsdGenAttributeMap::Sample Bilinear (attributeMap.cpp is
    // canonical): same checks, same clamp, same node math, same lerp
    // order, without the out-of-line call per root. The channel was
    // validated once above and value is never null, so those checks
    // drop out; Bilinear with res == 1 takes Sample's Nearest path.
    float const *texels = input.map->Data();
    int const numFaces = input.map->NumFaces();
    int const res = input.map->Resolution();
    int const channels = input.map->Channels();
    int const channel = input.channel;
    auto texel = [&](int face, int s, int t) -> float {
        s = std::min(res - 1, std::max(0, s));
        t = std::min(res - 1, std::max(0, t));
        return texels[((size_t(face) * size_t(res) + size_t(t)) * size_t(res) +
                        size_t(s)) *
                        size_t(channels) +
                    size_t(channel)];
    };
    // Bilinear address strides (bit-identical): texel(face, s, t) is
    // ((face * res + t) * res + s) * channels + channel; regrouped over
    // unsigned arithmetic (associative, no UB) as face * faceStride +
    // t * rowStride + s * channels + channel with faceStride ==
    // res * res * channels and rowStride == res * channels. Same index
    // for every in-range (face, s, t), two multiplies per corner
    // instead of three.
    size_t const faceStride =
        size_t(res) * size_t(res) * size_t(channels);
    size_t const rowStride = size_t(res) * size_t(channels);
    // Per-root sampler (bit-identical): fills values/keep/prototype for
    // root i and returns its slot (-1 when dropped). Every root reads
    // only its own input plus shared-immutable map/params and writes
    // only its own three output lanes, so any root order -- serial or
    // chunked over workers -- fills identical planes.
    auto sampleRoot = [&](size_t i) -> int {
        UsdGenAttributeInstanceRoot const &root = input.roots[i];
        float value = input.defaultValue;
        bool sampled = false;
        float u = root.u, v = root.v;
        if (root.face >= 0 && root.face < numFaces && std::isfinite(u) &&
            std::isfinite(v)) {
            // fminf/fmaxf (bit-identical): u/v just passed isfinite, and
            // for finite inputs fmin/fmax match std::min/max on every
            // value including signed zeros (fmax(+0,-0) = +0, the same
            // +0.0f std::max(0.0f, -0.0f) yields). Single instructions
            // replacing four NaN-conservative branches per root.
            u = fminf(1.0f, fmaxf(0.0f, u));
            v = fminf(1.0f, fmaxf(0.0f, v));
            if (res == 1) {
                int const s = std::min(
                    res - 1,
                    static_cast<int>(std::floor(double(u) * (res - 1) + 0.5)));
                int const t = std::min(
                    res - 1,
                    static_cast<int>(std::floor(double(v) * (res - 1) + 0.5)));
                value = texel(root.face, s, t);
            } else {
                // Redundant-clamp removal (bit-identical): u/v were clamped
                // to [0, 1] above, so x/y land in [0, res - 1] (rounding is
                // monotonic and 1.0 * k is exact), s0/t0 land in [0, res - 1]
                // (floor is exact), and x - s0 / y - t0 are Sterbenz-exact
                // fractions in [0, 1) whose [0, 1] clamp is the identity.
                // Only the +1 corners can reach res (iff u/v == 1.0 exactly),
                // so two predictable top clamps replace eight min/max pairs.
                double const x = double(u) * (res - 1);
                double const y = double(v) * (res - 1);
                int const s0 = static_cast<int>(std::floor(x));
                int const t0 = static_cast<int>(std::floor(y));
                double const fx = x - s0;
                double const fy = y - t0;
                int s1 = s0 + 1;
                if (s1 >= res) s1 = res - 1;
                int t1 = t0 + 1;
                if (t1 >= res) t1 = res - 1;
                // Corner-offset fetch (bit-identical): the +1 corners sit
                // one texel past their base corner (ds == 0 when clamped),
                // so two strided bases plus a shared step replace four
                // full address evaluations. Same four indices as texel.
                size_t const faceBase = size_t(root.face) * faceStride +
                                        size_t(channel);
                size_t const i00 = faceBase + size_t(t0) * rowStride +
                                   size_t(s0) * size_t(channels);
                size_t const i01 = faceBase + size_t(t1) * rowStride +
                                   size_t(s0) * size_t(channels);
                size_t const ds =
                    size_t(s1 - s0) * size_t(channels);
                double const v00 = texels[i00];
                double const v10 = texels[i00 + ds];
                double const v01 = texels[i01];
                double const v11 = texels[i01 + ds];
                value = static_cast<float>((v00 * (1.0 - fx) + v10 * fx) *
                                               (1.0 - fy) +
                                           (v01 * (1.0 - fx) + v11 * fx) * fy);
            }
            sampled = true;
        }
        if (!sampled) value = input.defaultValue;
        cooked.values[i] = value;
        bool const kept = sampled && value >= input.threshold;
        cooked.keep[i] = kept ? uint8_t(1) : uint8_t(0);
        if (!kept) {
            cooked.prototype[i] = -1;
            return -1;
        }
        // fminf/fmaxf (bit-identical): kept roots carry non-NaN values
        // (NaN fails the >= threshold above), and for non-NaN inputs
        // fmin/fmax match std::min/max exactly, infinities included.
        float const clamped = fminf(1.0f, fmaxf(0.0f, value));
        int slot = static_cast<int>(clamped * input.numPrototypes);
        if (slot >= input.numPrototypes) slot = input.numPrototypes - 1;
        cooked.prototype[i] = slot;
        return slot;
    };
    // Threaded sampling over root ranges for big cooks (plain TBB: the
    // cook API carries no scheduler context): each chunk samples its own
    // range into the shared planes (disjoint lanes) plus chunk-local
    // slot lists, which concatenate in chunk order -- chunk ranges are
    // ascending, so every slot's list matches the serial push order
    // exactly. kept is an order-free sum. Small cooks, single-worker
    // arenas, and huge prototype counts (chunk-local lists would fan
    // out) stay serial. The digest below runs serially either way over
    // the identical planes, so it is identical too.
    int const cookWorkers = tbb::this_task_arena::max_concurrency();
    size_t const cookChunks =
        (cookWorkers > 1 && count > 32768 && input.numPrototypes <= 4096)
            ? std::min({size_t(cookWorkers), size_t(8), count})
            : 1;
    if (cookChunks == 1) {
        for (size_t i = 0; i < count; ++i) {
            int const slot = sampleRoot(i);
            if (slot < 0) continue;
            cooked.instanceIndices[size_t(slot)].push_back(int(i));
            ++cooked.kept;
        }
    } else {
        struct CookChunk {
            std::vector<std::vector<int>> slots;
            size_t kept = 0;
        };
        std::vector<CookChunk> chunks(cookChunks);
        size_t const nProto = size_t(input.numPrototypes);
        size_t const perChunkShare = (perSlot + cookChunks - 1) / cookChunks;
        for (CookChunk &ch : chunks) {
            ch.slots.resize(nProto);
            for (std::vector<int> &slot : ch.slots)
                slot.reserve(perChunkShare);
        }
        tbb::parallel_for(tbb::blocked_range<size_t>(0, cookChunks),
            [&](tbb::blocked_range<size_t> const &range) {
                for (size_t c = range.begin(); c != range.end(); ++c) {
                    size_t const i0 = (c * count) / cookChunks;
                    size_t const i1 = ((c + 1) * count) / cookChunks;
                    CookChunk &ch = chunks[c];
                    for (size_t i = i0; i < i1; ++i) {
                        int const slot = sampleRoot(i);
                        if (slot < 0) continue;
                        ch.slots[size_t(slot)].push_back(int(i));
                        ++ch.kept;
                    }
                }
            });
        for (size_t s = 0; s < nProto; ++s) {
            std::vector<int> &dst = cooked.instanceIndices[s];
            for (CookChunk const &ch : chunks)
                dst.insert(dst.end(), ch.slots[s].begin(), ch.slots[s].end());
        }
        for (CookChunk const &ch : chunks)
            cooked.kept += ch.kept;
    }

    // FNV-1a over the map digest, the parameters and every assignment.
    uint64_t hash = 14695981039346656037ull;
    UsdGenDigestMixWord(hash, input.map->Digest());
    UsdGenDigestMixWord(hash, uint64_t(input.channel));
    UsdGenDigestMixWord(hash, uint64_t(input.numPrototypes));
    UsdGenDigestMixWord(hash, uint64_t(count));
    uint32_t bits = 0;
    std::memcpy(&bits, &input.threshold, sizeof(bits));
    UsdGenDigestMixWord(hash, bits);
    std::memcpy(&bits, &input.defaultValue, sizeof(bits));
    UsdGenDigestMixWord(hash, bits);
    // Per-root assignments through 4 FNV lanes over strided roots (see
    // digest.h): the same words feed each lane in root order, so every
    // assignment stays digest-sensitive while the four chains overlap.
    // Tail roots join lane 0. Lane seeds domain-separate the header hash.
    // One word per root: keep is redundant with the prototype (kept roots
    // carry a slot, dropped roots carry -1), so the value bits and the
    // prototype pack into a single feed. Digest values are internal keys
    // (digest.h), never persisted: the packing changes them by design.
    uint64_t lane[4] = {hash, hash ^ 0x9E3779B97F4A7C15ull,
                        hash ^ 0xBF58476D1CE4E5B9ull,
                        hash ^ 0x94D049BB133111EBull};
    auto feed = [&](uint64_t &h, size_t i) {
        uint32_t vbits = 0;
        std::memcpy(&vbits, &cooked.values[i], sizeof(vbits));
        uint64_t const packed = (uint64_t(vbits) << 32) |
                                uint64_t(uint32_t(cooked.prototype[i]));
        UsdGenDigestMixWord(h, packed);
    };
    // Threaded over lanes for big cooks (same rule as the sampling):
    // the four lanes are independent chains over disjoint strided word
    // sets, so each lane runs on its own worker with the identical word
    // sequence -- lane states and the combine are bit-identical. The
    // <=3 tail roots join lane 0 serially after the join, as before.
    size_t const n4 = count & ~size_t(3);
    if (cookChunks == 1) {
        size_t i = 0;
        for (; i < n4; i += 4) {
            feed(lane[0], i + 0);
            feed(lane[1], i + 1);
            feed(lane[2], i + 2);
            feed(lane[3], i + 3);
        }
        for (; i < count; ++i)
            feed(lane[0], i);
    } else {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, 4),
            [&](tbb::blocked_range<size_t> const &range) {
                for (size_t l = range.begin(); l != range.end(); ++l) {
                    uint64_t h = lane[l];
                    for (size_t i = l; i < n4; i += 4)
                        feed(h, i);
                    lane[l] = h;
                }
            });
        for (size_t i = n4; i < count; ++i)
            feed(lane[0], i);
    }
    cooked.digest = UsdGenDigestCombine4(lane[0], lane[1], lane[2], lane[3]);

    *result = std::move(cooked);
    return true;
}

}  // namespace usdGen
