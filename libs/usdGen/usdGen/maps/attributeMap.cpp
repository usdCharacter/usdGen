// usdGen — paintable per-surface attribute maps and stroke accumulation.
// See attributeMap.h for the model and its v1 limits.
#include "usdGen/maps/attributeMap.h"

#include "usdGen/digest.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace usdGen {

namespace {

// 2^28 floats (1 GiB) of texels: far past any real scalp map, and the guard
// that keeps a corrupt spec from sizing a hostile allocation.
constexpr int64_t kMaxFloats = int64_t(1) << 28;

bool IsPowerOfTwo(int value)
{
    return value > 0 && (value & (value - 1)) == 0;
}

float Clamp01f(float value)
{
    return std::min(1.0f, std::max(0.0f, value));
}

float FalloffCurve(UsdGenBrushFalloff falloff, double t)
{
    // t runs 1 at the inner radius to 0 at the rim.
    switch (falloff) {
    case UsdGenBrushFalloff::Constant: return 1.0f;
    case UsdGenBrushFalloff::Linear: return static_cast<float>(t);
    case UsdGenBrushFalloff::Smooth: break;
    }
    float const x = static_cast<float>(t);
    return x * x * (3.0f - 2.0f * x);
}

}  // namespace

float UsdGenBrushWeight(UsdGenBrushFalloff falloff, float hardness, double dist,
                        double radius)
{
    if (!(radius > 0.0) || !(dist <= radius)) return 0.0f;
    double const h = std::min(1.0, std::max(0.0, double(hardness)));
    double const inner = h * radius;
    if (dist <= inner) return 1.0f;
    double const span = radius - inner;
    if (!(span > 0.0)) return 1.0f;
    double const t = std::min(1.0, std::max(0.0, 1.0 - (dist - inner) / span));
    return FalloffCurve(falloff, t);
}

// ---------------------------------------------------------------------------
// UsdGenAttributeMap
// ---------------------------------------------------------------------------

struct UsdGenAttributeMap::State {
    UsdGenAttributeMapSpec spec;
    std::vector<float> texels;
    // Cached Digest(): the full-texel hash costs milliseconds on large
    // maps and cooks rarely follow a mutation, so Digest memoizes until
    // the next write. EVERY texel write must clear digestValid: SetTexel
    // and Fill do it directly, MutableData dirties eagerly on fetch since
    // bulk writes bypass per-texel hooks. Atomics keep concurrent const
    // Digest() calls race-free (mutation-vs-access was already exclusive).
    std::atomic<bool> digestValid{false};
    std::atomic<uint64_t> digestCache{0};
};

UsdGenAttributeMap::UsdGenAttributeMap() : state_(std::make_unique<State>()) {}

UsdGenAttributeMap::~UsdGenAttributeMap() = default;

std::shared_ptr<UsdGenAttributeMap> UsdGenAttributeMap::Create(
    UsdGenAttributeMapSpec const &spec, std::string *error)
{
    auto fail = [&](std::string const &message) -> std::shared_ptr<UsdGenAttributeMap> {
        if (error) *error = message;
        return nullptr;
    };
    if (spec.numFaces <= 0)
        return fail("attribute map needs numFaces >= 1");
    if (!IsPowerOfTwo(spec.resolution) || spec.resolution < 1 || spec.resolution > 256)
        return fail("attribute map resolution must be a power of two in [1, 256]");
    if (spec.channels != 1 && spec.channels != 3)
        return fail("attribute map channels must be 1 or 3");
    if (!std::isfinite(spec.defaultValue))
        return fail("attribute map default must be finite");
    int64_t const floats =
        int64_t(spec.numFaces) * spec.resolution * spec.resolution * spec.channels;
    if (floats > kMaxFloats)
        return fail("attribute map grid exceeds 2^28 floats");
    auto map = std::shared_ptr<UsdGenAttributeMap>(new UsdGenAttributeMap());
    map->state_->spec = spec;
    if (spec.clamp01) map->state_->spec.defaultValue = Clamp01f(spec.defaultValue);
    map->state_->texels.assign(size_t(floats), map->state_->spec.defaultValue);
    return map;
}

int UsdGenAttributeMap::NumFaces() const { return state_->spec.numFaces; }
int UsdGenAttributeMap::Resolution() const { return state_->spec.resolution; }
int UsdGenAttributeMap::Channels() const { return state_->spec.channels; }
float UsdGenAttributeMap::DefaultValue() const { return state_->spec.defaultValue; }
bool UsdGenAttributeMap::Clamp01() const { return state_->spec.clamp01; }

std::shared_ptr<UsdGenAttributeMap> UsdGenAttributeMap::Clone() const
{
    auto map = std::shared_ptr<UsdGenAttributeMap>(new UsdGenAttributeMap());
    map->state_->spec = state_->spec;
    map->state_->texels = state_->texels;
    return map;
}

bool UsdGenAttributeMap::GetTexel(int face, int s, int t, int channel, float *value) const
{
    State const &state = *state_;
    if (!value) return false;
    if (face < 0 || face >= state.spec.numFaces) return false;
    if (s < 0 || s >= state.spec.resolution || t < 0 || t >= state.spec.resolution) return false;
    if (channel < 0 || channel >= state.spec.channels) return false;
    size_t const index = ((size_t(face) * size_t(state.spec.resolution) + size_t(t)) *
                              size_t(state.spec.resolution) +
                          size_t(s)) *
                             size_t(state.spec.channels) +
                         size_t(channel);
    *value = state.texels[index];
    return true;
}

bool UsdGenAttributeMap::SetTexel(int face, int s, int t, int channel, float value)
{
    if (!std::isfinite(value)) return false;
    State &state = *state_;
    if (face < 0 || face >= state.spec.numFaces) return false;
    if (s < 0 || s >= state.spec.resolution || t < 0 || t >= state.spec.resolution) return false;
    if (channel < 0 || channel >= state.spec.channels) return false;
    if (state.spec.clamp01) value = Clamp01f(value);
    size_t const index = ((size_t(face) * size_t(state.spec.resolution) + size_t(t)) *
                              size_t(state.spec.resolution) +
                          size_t(s)) *
                             size_t(state.spec.channels) +
                         size_t(channel);
    state.texels[index] = value;
    state.digestValid.store(false, std::memory_order_relaxed);
    return true;
}

bool UsdGenAttributeMap::Sample(int face, float u, float v, int channel,
                                UsdGenAttributeMapInterp interp, float *value) const
{
    State const &state = *state_;
    if (!value) return false;
    if (face < 0 || face >= state.spec.numFaces) return false;
    if (channel < 0 || channel >= state.spec.channels) return false;
    if (!std::isfinite(u) || !std::isfinite(v)) return false;
    u = Clamp01f(u);
    v = Clamp01f(v);
    int const res = state.spec.resolution;
    auto texel = [&](int s, int t) -> float {
        s = std::min(res - 1, std::max(0, s));
        t = std::min(res - 1, std::max(0, t));
        return state
            .texels[((size_t(face) * size_t(res) + size_t(t)) * size_t(res) + size_t(s)) *
                        size_t(state.spec.channels) +
                    size_t(channel)];
    };
    if (interp == UsdGenAttributeMapInterp::Nearest || res == 1) {
        int const s = std::min(
            res - 1, static_cast<int>(std::floor(double(u) * (res - 1) + 0.5)));
        int const t = std::min(
            res - 1, static_cast<int>(std::floor(double(v) * (res - 1) + 0.5)));
        *value = texel(s, t);
        return true;
    }
    // Bilinear over grid nodes at s / (res - 1): exact at the corners
    // with no edge-clamp collapse, exact between nodes. Mirrors the
    // tool's BrushMap.sample (brushMap.py) and the base upsample.
    double const x = double(u) * (res - 1);
    double const y = double(v) * (res - 1);
    int const s0 = static_cast<int>(std::floor(x));
    int const t0 = static_cast<int>(std::floor(y));
    double const fx = std::min(1.0, std::max(0.0, x - s0));
    double const fy = std::min(1.0, std::max(0.0, y - t0));
    double const v00 = texel(s0, t0);
    double const v10 = texel(s0 + 1, t0);
    double const v01 = texel(s0, t0 + 1);
    double const v11 = texel(s0 + 1, t0 + 1);
    *value = static_cast<float>((v00 * (1.0 - fx) + v10 * fx) * (1.0 - fy) +
                                (v01 * (1.0 - fx) + v11 * fx) * fy);
    return true;
}

void UsdGenAttributeMap::Fill(float value)
{
    if (!std::isfinite(value)) return;
    if (state_->spec.clamp01) value = Clamp01f(value);
    std::fill(state_->texels.begin(), state_->texels.end(), value);
    state_->digestValid.store(false, std::memory_order_relaxed);
}

uint64_t UsdGenAttributeMap::Digest() const
{
    if (state_->digestValid.load(std::memory_order_acquire))
        return state_->digestCache.load(std::memory_order_relaxed);
    // 4-lane FNV-1a (see digest.h) over the spec words then every texel
    // bit: the spec feeds the single-lane header, whose hash seeds the
    // bulk texel lanes. Same contract — bitwise-identical maps digest
    // identically — at ~4x the throughput; values are internal keys.
    uint64_t hash = 14695981039346656037ull;
    UsdGenDigestMixWord(hash, uint64_t(state_->spec.numFaces));
    UsdGenDigestMixWord(hash, uint64_t(state_->spec.resolution));
    UsdGenDigestMixWord(hash, uint64_t(state_->spec.channels));
    UsdGenDigestMixWord(hash, uint64_t(state_->spec.clamp01 ? 1 : 0));
    uint32_t defaultBits = 0;
    std::memcpy(&defaultBits, &state_->spec.defaultValue, sizeof(defaultBits));
    UsdGenDigestMixWord(hash, defaultBits);
    static_assert(sizeof(float) == 4, "float is 32 bits");
    uint64_t const digest = UsdGenDigestBytes(state_->texels.data(),
                                              state_->texels.size() * sizeof(float), hash);
    state_->digestCache.store(digest, std::memory_order_relaxed);
    state_->digestValid.store(true, std::memory_order_release);
    return digest;
}

float const *UsdGenAttributeMap::Data() const { return state_->texels.data(); }
size_t UsdGenAttributeMap::FloatCount() const { return state_->texels.size(); }
float *UsdGenAttributeMap::MutableData()
{
    // Eager digest invalidation: bulk writes bypass per-texel hooks, so
    // any fetch dirties the cache. Complete all writes through the
    // pointer before the next Digest(); holding it across a Digest call
    // with further writes would read back a stale epoch key.
    state_->digestValid.store(false, std::memory_order_relaxed);
    return state_->texels.data();
}

// ---------------------------------------------------------------------------
// UsdGenBrushStroke
// ---------------------------------------------------------------------------

struct UsdGenBrushStroke::State {
    std::shared_ptr<const UsdGenAttributeMap> base;
    std::vector<UsdGenBrushDab> dabs;
};

UsdGenBrushStroke::UsdGenBrushStroke(std::shared_ptr<const UsdGenAttributeMap> base)
    : state_(std::make_unique<State>())
{
    state_->base = std::move(base);
}

UsdGenBrushStroke::~UsdGenBrushStroke() = default;

bool UsdGenBrushStroke::AddDab(UsdGenBrushDab const &dab, std::string *error)
{
    auto fail = [&](std::string const &message) {
        if (error) *error = message;
        return false;
    };
    if (!state_->base) return fail("brush stroke has no base map");
    if (dab.face < 0 || dab.face >= state_->base->NumFaces())
        return fail("brush dab face " + std::to_string(dab.face) + " is outside the base map");
    if (!std::isfinite(dab.u) || !std::isfinite(dab.v))
        return fail("brush dab (u, v) must be finite");
    if (!std::isfinite(dab.radius) || dab.radius < 0.0f)
        return fail("brush dab radius must be finite and >= 0");
    if (!std::isfinite(dab.strength) || dab.strength < 0.0f || dab.strength > 1.0f)
        return fail("brush dab strength must be finite and in [0, 1]");
    if (!std::isfinite(dab.value)) return fail("brush dab value must be finite");
    if (!std::isfinite(dab.hardness) || dab.hardness < 0.0f || dab.hardness > 1.0f)
        return fail("brush dab hardness must be finite and in [0, 1]");
    if (dab.channel < -1 || dab.channel >= state_->base->Channels())
        return fail("brush dab channel is outside the base map");
    state_->dabs.push_back(dab);
    return true;
}

bool UsdGenBrushStroke::AddMove(int face, float u, float v, float radius, float strength,
                                float value, int channel, UsdGenBrushMode mode,
                                UsdGenBrushFalloff falloff, float spacing, std::string *error)
{
    UsdGenBrushDab dab;
    dab.face = face;
    dab.u = u;
    dab.v = v;
    dab.radius = radius;
    dab.strength = strength;
    dab.value = value;
    dab.channel = channel;
    dab.mode = mode;
    dab.falloff = falloff;
    return AddMove(dab, spacing, error);
}

bool UsdGenBrushStroke::AddMove(UsdGenBrushDab const &dab, float spacing, std::string *error)
{
    auto fail = [&](std::string const &message) {
        if (error) *error = message;
        return false;
    };
    if (!std::isfinite(spacing) || spacing <= 0.0f || spacing > 1.0f)
        return fail("brush move spacing must be finite and in (0, 1]");
    // Validate the endpoint before touching the dab list: a rejected move
    // records nothing.
    {
        UsdGenBrushStroke probe(state_->base);
        if (!probe.AddDab(dab, error)) return false;
    }
    float const u = dab.u;
    float const v = dab.v;
    float const radius = dab.radius;
    bool interpolate = !state_->dabs.empty() && state_->dabs.back().face == dab.face &&
                       radius > 0.0f && std::isfinite(u) && std::isfinite(v);
    if (!interpolate) {
        state_->dabs.push_back(dab);
        return true;
    }
    UsdGenBrushDab const prev = state_->dabs.back();
    double const dx = double(u) - prev.u;
    double const dy = double(v) - prev.v;
    double const dist = std::sqrt(dx * dx + dy * dy);
    double const step = double(spacing) * radius;
    // The start point is the previous dab's centre, already painted: the
    // segment's interior plus the endpoint are what get stamped.
    int64_t steps = step > 0.0 ? int64_t(std::ceil(dist / step)) : 1;
    if (steps < 1) steps = 1;
    if (steps > 1024) steps = 1024;  // a wild move stamps a bounded trail
    for (int64_t i = 1; i <= steps; ++i) {
        double const t = double(i) / double(steps);
        UsdGenBrushDab stamp = dab;
        stamp.u = static_cast<float>(prev.u + dx * t);
        stamp.v = static_cast<float>(prev.v + dy * t);
        state_->dabs.push_back(stamp);
    }
    return true;
}

size_t UsdGenBrushStroke::DabCount() const { return state_->dabs.size(); }

UsdGenBrushDab const &UsdGenBrushStroke::Dab(size_t index) const
{
    return state_->dabs[index];
}

// Applies one dab to a live map: every texel whose grid node falls within
// the radius (face-uv units) is visited exactly once, in (t, s) order, so
// the result is deterministic for a given dab list.
void UsdGenApplyBrushDab(UsdGenAttributeMap *map, UsdGenBrushDab const &dabIn)
{
    if (!map) return;
    UsdGenBrushDab dab = dabIn;
    if (dab.mode == UsdGenBrushMode::Erase) {
        // Erase is Set toward the map default.
        dab.mode = UsdGenBrushMode::Set;
        dab.value = map->DefaultValue();
    }
    int const res = map->Resolution();
    int const channels = map->Channels();
    int const first = dab.channel < 0 ? 0 : dab.channel;
    int const last = dab.channel < 0 ? channels : dab.channel + 1;
    if (dab.strength == 0.0f) return;

    if (!(dab.radius > 0.0f)) {
        int const s = std::min(
            res - 1, std::max(0, static_cast<int>(std::floor(double(dab.u) * (res - 1) + 0.5))));
        int const t = std::min(
            res - 1, std::max(0, static_cast<int>(std::floor(double(dab.v) * (res - 1) + 0.5))));
        for (int c = first; c < last; ++c) {
            float current = 0.0f;
            if (!map->GetTexel(dab.face, s, t, c, &current)) continue;
            float next = current;
            if (dab.mode == UsdGenBrushMode::Set) {
                next = current + (dab.value - current) * dab.strength;
            } else if (dab.mode == UsdGenBrushMode::Add) {
                next = current + dab.value * dab.strength;
            } else {
                // Smooth of a single texel with no footprint is the identity:
                // there is no neighbourhood to relax toward.
                continue;
            }
            map->SetTexel(dab.face, s, t, c, next);
        }
        return;
    }

    double const radius = dab.radius;
    int const span = res > 1 ? res - 1 : 1;
    int const sLo = std::max(0, static_cast<int>(std::floor((dab.u - radius) * span)));
    int const sHi = std::min(res - 1, static_cast<int>(std::ceil((dab.u + radius) * span)));
    int const tLo = std::max(0, static_cast<int>(std::floor((dab.v - radius) * span)));
    int const tHi = std::min(res - 1, static_cast<int>(std::ceil((dab.v + radius) * span)));

    // Smooth reads the neighbourhood mean from the pre-dab map. The mean per
    // footprint texel is folded once, up front, over the footprint plus its
    // one-texel ring (nine lookups per texel in the hot loop costs the drag
    // its budget); the summation order matches the per-texel fold bit for
    // bit. Set/Add read-modify-write in place. A footprint entirely past
    // the face touches no texel in any mode.
    if (sLo > sHi || tLo > tHi) return;
    int const snapS0 = std::max(0, sLo - 1);
    int const snapT0 = std::max(0, tLo - 1);
    int const snapS1 = std::min(res - 1, sHi + 1);
    int const snapT1 = std::min(res - 1, tHi + 1);
    size_t const snapW = size_t(snapS1 - snapS0 + 1);
    size_t const footW = size_t(sHi - sLo + 1);
    std::vector<double> means;
    if (dab.mode == UsdGenBrushMode::Smooth) {
        std::vector<float> snap(size_t(channels) * size_t(snapT1 - snapT0 + 1) * snapW, 0.0f);
        for (int t = snapT0; t <= snapT1; ++t) {
            for (int s = snapS0; s <= snapS1; ++s) {
                for (int c = first; c < last; ++c) {
                    float current = 0.0f;
                    map->GetTexel(dab.face, s, t, c, &current);
                    snap[(size_t(t - snapT0) * snapW + size_t(s - snapS0)) * size_t(channels) +
                         size_t(c)] = current;
                }
            }
        }
        auto snapAt = [&](int s, int t, int c) -> float {
            s = std::min(res - 1, std::max(0, s));
            t = std::min(res - 1, std::max(0, t));
            return snap[(size_t(t - snapT0) * snapW + size_t(s - snapS0)) * size_t(channels) +
                        size_t(c)];
        };
        means.assign(size_t(channels) * size_t(tHi - tLo + 1) * footW, 0.0f);
        for (int t = tLo; t <= tHi; ++t) {
            for (int s = sLo; s <= sHi; ++s) {
                for (int c = first; c < last; ++c) {
                    double mean = 0.0;
                    for (int dt = -1; dt <= 1; ++dt)
                        for (int ds = -1; ds <= 1; ++ds) mean += snapAt(s + ds, t + dt, c);
                    mean /= 9.0;
                    means[(size_t(t - tLo) * footW + size_t(s - sLo)) * size_t(channels) +
                          size_t(c)] = mean;
                }
            }
        }
    }

    for (int t = tLo; t <= tHi; ++t) {
        for (int s = sLo; s <= sHi; ++s) {
            double const cu = res > 1 ? double(s) / span : 0.5;
            double const cv = res > 1 ? double(t) / span : 0.5;
            double const dist = std::sqrt((cu - dab.u) * (cu - dab.u) + (cv - dab.v) * (cv - dab.v));
            if (dist > radius) continue;
            float const weight = UsdGenBrushWeight(dab.falloff, dab.hardness, dist, radius);
            float const k = dab.strength * weight;
            if (k == 0.0f) continue;
            for (int c = first; c < last; ++c) {
                float current = 0.0f;
                if (!map->GetTexel(dab.face, s, t, c, &current)) continue;
                float next = current;
                if (dab.mode == UsdGenBrushMode::Set) {
                    next = current + (dab.value - current) * k;
                } else if (dab.mode == UsdGenBrushMode::Add) {
                    next = current + dab.value * k;
                } else {
                    double const mean = means[(size_t(t - tLo) * footW + size_t(s - sLo)) *
                                              size_t(channels) +
                                        size_t(c)];
                    next = static_cast<float>(current + (mean - current) * k);
                }
                map->SetTexel(dab.face, s, t, c, next);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// UsdGenBrushAccumulator
// ---------------------------------------------------------------------------

struct UsdGenBrushAccumulator::State {
    UsdGenAttributeMap *map = nullptr;
    bool active = false;
    UsdGenBrushMode mode = UsdGenBrushMode::Set;
    uint32_t targetBits = 0;
    int channel = -1;
    std::shared_ptr<UsdGenAttributeMap> base;  // the segment's start
    std::vector<float> wmax;                   // one per texel (all channels)
};

UsdGenBrushAccumulator::UsdGenBrushAccumulator(UsdGenAttributeMap *map)
    : state_(std::make_unique<State>())
{
    state_->map = map;
}

UsdGenBrushAccumulator::~UsdGenBrushAccumulator() = default;

void UsdGenBrushAccumulator::Flush()
{
    state_->active = false;
    state_->base.reset();
    state_->wmax.clear();
}

void UsdGenBrushAccumulator::Apply(UsdGenBrushDab const &dab)
{
    State &st = *state_;
    UsdGenAttributeMap *map = st.map;
    if (!map) return;
    if (dab.mode == UsdGenBrushMode::Add) {
        // Add accumulates per stamp by design; it ends any max segment.
        Flush();
        UsdGenApplyBrushDab(map, dab);
        return;
    }
    if (dab.strength == 0.0f) return;
    if (dab.face < 0 || dab.face >= map->NumFaces()) return;
    float const target = dab.mode == UsdGenBrushMode::Erase ? map->DefaultValue() : dab.value;
    uint32_t bits = 0;
    if (dab.mode != UsdGenBrushMode::Smooth) std::memcpy(&bits, &target, sizeof(bits));
    if (!st.active || st.mode != dab.mode || st.targetBits != bits || st.channel != dab.channel) {
        Flush();
        st.active = true;
        st.mode = dab.mode;
        st.targetBits = bits;
        st.channel = dab.channel;
        st.base = map->Clone();
        st.wmax.assign(size_t(map->NumFaces()) * size_t(map->Resolution()) *
                           size_t(map->Resolution()),
                       0.0f);
    }
    int const res = map->Resolution();
    int const channels = map->Channels();
    int const first = dab.channel < 0 ? 0 : dab.channel;
    int const last = dab.channel < 0 ? channels : dab.channel + 1;
    float const *base = st.base->Data();
    float *data = map->MutableData();
    bool const clamp = map->Clamp01();
    auto texelIndex = [&](int s, int t) {
        return (size_t(dab.face) * size_t(res) + size_t(t)) * size_t(res) + size_t(s);
    };
    auto baseAt = [&](int s, int t, int c) {
        s = std::min(res - 1, std::max(0, s));
        t = std::min(res - 1, std::max(0, t));
        return base[texelIndex(s, t) * size_t(channels) + size_t(c)];
    };
    auto stamp = [&](int s, int t, float w) {
        size_t const i = texelIndex(s, t);
        if (!(w > st.wmax[i])) return;
        st.wmax[i] = w;
        for (int c = first; c < last; ++c) {
            double const b = baseAt(s, t, c);
            double goal = target;
            if (st.mode == UsdGenBrushMode::Smooth) {
                double mean = 0.0;
                for (int dt = -1; dt <= 1; ++dt)
                    for (int ds = -1; ds <= 1; ++ds) mean += baseAt(s + ds, t + dt, c);
                goal = mean / 9.0;
            }
            float v = static_cast<float>(b + (goal - b) * double(w));
            if (clamp) v = Clamp01f(v);
            data[i * size_t(channels) + size_t(c)] = v;
        }
    };
    if (!(dab.radius > 0.0f)) {
        // A point dab: the nearest texel at full weight (Smooth of a single
        // texel with no footprint stays the identity).
        if (st.mode == UsdGenBrushMode::Smooth) return;
        int const s = std::min(
            res - 1, std::max(0, static_cast<int>(std::floor(double(dab.u) * (res - 1) + 0.5))));
        int const t = std::min(
            res - 1, std::max(0, static_cast<int>(std::floor(double(dab.v) * (res - 1) + 0.5))));
        stamp(s, t, dab.strength);
        return;
    }
    double const radius = dab.radius;
    int const span = res > 1 ? res - 1 : 1;
    int const sLo = std::max(0, static_cast<int>(std::floor((dab.u - radius) * span)));
    int const sHi = std::min(res - 1, static_cast<int>(std::ceil((dab.u + radius) * span)));
    int const tLo = std::max(0, static_cast<int>(std::floor((dab.v - radius) * span)));
    int const tHi = std::min(res - 1, static_cast<int>(std::ceil((dab.v + radius) * span)));
    for (int t = tLo; t <= tHi; ++t) {
        for (int s = sLo; s <= sHi; ++s) {
            double const cu = res > 1 ? double(s) / span : 0.5;
            double const cv = res > 1 ? double(t) / span : 0.5;
            double const dist = std::sqrt((cu - dab.u) * (cu - dab.u) + (cv - dab.v) * (cv - dab.v));
            if (dist > radius) continue;
            float const w = dab.strength * UsdGenBrushWeight(dab.falloff, dab.hardness, dist, radius);
            if (w > 0.0f) stamp(s, t, w);
        }
    }
}

std::shared_ptr<UsdGenAttributeMap> UsdGenBrushStroke::Preview() const
{
    if (!state_->base) return nullptr;
    std::shared_ptr<UsdGenAttributeMap> map = state_->base->Clone();
    UsdGenBrushAccumulator accumulator(map.get());
    for (UsdGenBrushDab const &dab : state_->dabs) accumulator.Apply(dab);
    return map;
}

std::shared_ptr<UsdGenAttributeMap> UsdGenBrushStroke::Commit() const { return Preview(); }

void UsdGenBrushStroke::Abort() { state_->dabs.clear(); }

}  // namespace usdGen
