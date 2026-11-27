#include "usdGenCurves/cuda/length.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenCuda/gpu/cudaCompat.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>

namespace usdGen::gpu {
namespace {

constexpr int kBadOffsets = 1;
constexpr int kNonFinite = 2;
constexpr int kBadValue = 3;
constexpr uint32_t kSaltLength = 0x4C656E67u;

__device__ void SetError(int *error, int value) { atomicCAS(error, 0, value); }
__device__ int LoadError(const int *error) {
    return atomicAdd(const_cast<int *>(error), 0);
}
__device__ bool Finite(float3 p) {
    return isfinite(p.x) && isfinite(p.y) && isfinite(p.z);
}

__device__ uint64_t Hash64(uint64_t key, uint32_t salt) {
    uint64_t z = (key ^ (uint64_t(salt) * 0x9E3779B97F4A7C15ull)) +
                 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ float Draw01(int seed, uint64_t id) {
    const uint64_t key = Hash64(uint64_t(uint32_t(seed)), kSaltLength) ^ id;
    return float(uint32_t(Hash64(key, kSaltLength) >> 32) >> 8) * 0x1.0p-24f;
}

__device__ bool ReadScalar(ScalarField f, size_t curve, size_t point,
                           float *value, int *error) {
    size_t index = 0;
    if (f.data) {
        if (f.domain == expr::Domain::Primitive) index = curve;
        else if (f.domain == expr::Domain::Point) index = point;
        else if (f.domain != expr::Domain::Groom) {
            SetError(error, kBadValue); return false;
        }
        if (index >= f.count) { SetError(error, kBadValue); return false; }
        *value = f.data[index];
    } else *value = f.literal;
    if (!isfinite(*value)) { SetError(error, kNonFinite); return false; }
    return true;
}

__device__ bool ReadVec2(Vec2Field f, size_t curve, size_t point,
                         float2 *value, int *error) {
    size_t index = 0;
    if (f.data) {
        if (f.domain == expr::Domain::Primitive) index = curve;
        else if (f.domain == expr::Domain::Point) index = point;
        else if (f.domain != expr::Domain::Groom) {
            SetError(error, kBadValue); return false;
        }
        if (index >= f.count) { SetError(error, kBadValue); return false; }
        *value = f.data[index];
    } else *value = f.literal;
    if (!isfinite(value->x) || !isfinite(value->y)) {
        SetError(error, kNonFinite); return false;
    }
    return true;
}

__device__ bool ReadBool(BoolField f, size_t curve, bool *value, int *error) {
    if (!f.data) { *value = f.literal; return true; }
    size_t index = 0;
    if (f.domain == expr::Domain::Primitive) index = curve;
    else if (f.domain == expr::Domain::Groom) {
        if (f.count != 1) { SetError(error, kBadValue); return false; }
    } else { SetError(error, kBadValue); return false; }
    if (index >= f.count || f.data[index] > 1) {
        SetError(error, kBadValue); return false;
    }
    *value = f.data[index] != 0;
    return true;
}

__device__ float HairT(const float *hairT, uint32_t point,
                       uint32_t begin, uint32_t end, int *error) {
    const float denom = float(end - begin - 1u);
    float t = denom > 0.0f ? float(point - begin) / denom : 0.0f;
    if (hairT) t = hairT[point];
    if (!isfinite(t) || t < 0.0f || t > 1.0f) {
        SetError(error, !isfinite(t) ? kNonFinite : kBadValue); return 0.0f;
    }
    return t;
}

__device__ float3 SampleArc(const float3 *points, uint32_t begin,
                            uint32_t end, float distance) {
    if (end <= begin + 1u) return points[begin];
    float remaining = distance;
    for (uint32_t i = begin + 1u; i < end; ++i) {
        const float3 a = points[i - 1u], b = points[i];
        const float3 d = make_float3(b.x - a.x, b.y - a.y, b.z - a.z);
        const float segment = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
        if (remaining <= segment || i + 1u == end) {
            const float u = segment > 0.0f
                ? fminf(1.0f, fmaxf(0.0f, remaining / segment)) : 0.0f;
            return make_float3(a.x + d.x*u, a.y + d.y*u, a.z + d.z*u);
        }
        remaining -= segment;
    }
    return points[end - 1u];
}

// This launch is deliberately separate from LengthKernel. No curve kernel
// reads an offset-derived point index until endpoint and monotonicity checks
// have completed on the preceding launch.
__global__ void ValidateKernel(DeviceCurveGeometryView g,
                               DeviceView<const float> hairT, int *error) {
    const size_t first = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x)*blockDim.x;
    for (size_t i = first; i <= g.curveCount; i += stride) {
        const uint32_t offset = g.curveOffsets.data[i];
        if ((i == 0 && offset != 0u) ||
            (i == g.curveCount && offset != g.pointCount) ||
            (i > 0 && i < g.curveCount && offset <= g.curveOffsets.data[i-1u]))
            SetError(error, kBadOffsets);
        if (i < g.curveCount && offset >= g.curveOffsets.data[i+1u])
            SetError(error, kBadOffsets);
    }
    for (size_t i = first; i < g.pointCount; i += stride) {
        if (!Finite(g.points.data[i])) SetError(error, kNonFinite);
        if (hairT.data && (!isfinite(hairT.data[i]) || hairT.data[i] < 0 ||
                           hairT.data[i] > 1))
            SetError(error, isfinite(hairT.data[i]) ? kBadValue : kNonFinite);
    }
}

__global__ void LengthKernel(DeviceCurveGeometryView g,
                             DeviceView<const float> hairT,
                             LengthParameters p, DeviceView<float3> output,
                             DeviceView<uint8_t> keep, int *error) {
    const size_t first = size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x)*blockDim.x;
    for (size_t c = first; c < g.curveCount; c += stride) {
        if (LoadError(error) != 0) continue;
        const uint32_t b = g.curveOffsets.data[c];
        const uint32_t e = g.curveOffsets.data[c + 1u];
        const float3 root = g.points.data[b];
        float current = 0;
        for (uint32_t i = b + 1u; i < e; ++i) {
            const float3 a = g.points.data[i-1u], q = g.points.data[i];
            const float3 d = make_float3(q.x-a.x, q.y-a.y, q.z-a.z);
            current += sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
        }
        if (!isfinite(current)) { SetError(error, kNonFinite); continue; }

        float2 randomRange;
        float rootValue = 0, rootMin = 0, threshold = 0;
        if (!ReadVec2(p.random, c, b, &randomRange, error) ||
            !ReadScalar(p.value, c, b, &rootValue, error) ||
            !ReadScalar(p.minRemainingLength, c, b, &rootMin, error) ||
            !ReadScalar(p.cullThreshold, c, b, &threshold, error)) continue;
        if (randomRange.x < 0 || randomRange.y < 0 || rootValue < 0 ||
            rootMin < 0 || threshold < 0) {
            SetError(error, kBadValue); continue;
        }
        if (randomRange.y < randomRange.x) {
            const float tmp = randomRange.x;
            randomRange.x = randomRange.y;
            randomRange.y = tmp;
        }
        const float multiplier = randomRange.x + (randomRange.y-randomRange.x) *
                                 Draw01(p.seed, g.stableIds.data ?
                                        g.stableIds.data[c] : uint64_t(c));
        if (!isfinite(multiplier)) { SetError(error, kNonFinite); continue; }

        bool enabled = true;
        if (!ReadBool(p.enabled, c, &enabled, error)) continue;
        if (p.mode == LengthMode::Cull) {
            bool allZeroEnvelope = true;
            for (uint32_t i = b; i < e; ++i) {
                float amount = 0.0f;
                if (!ReadScalar(p.mask, c, i, &amount, error) ||
                    amount < 0.0f || amount > 1.0f) {
                    SetError(error, kBadValue);
                    allZeroEnvelope = false;
                    break;
                }
                if (amount != 0.0f) allZeroEnvelope = false;
            }
            for (uint32_t i = b; i < e; ++i) output.data[i] = g.points.data[i];
            // Disabled controls are an exact no-op, including topology.
            keep.data[c] = !enabled || allZeroEnvelope ||
                          current >= threshold ? 1 : 0;
            continue;
        }
        if (!enabled) {
            for (uint32_t i = b; i < e; ++i) output.data[i] = g.points.data[i];
            keep.data[c] = 1;
            continue;
        }

        bool valid = true;
        bool allZeroEnvelope = true;
        for (uint32_t i = b; i < e; ++i) {
            float value = 0, amount = 0, minLength = 0;
            if (!ReadScalar(p.value, c, i, &value, error) ||
                !ReadScalar(p.mask, c, i, &amount, error) ||
                !ReadScalar(p.minRemainingLength, c, i, &minLength, error)) {
                valid = false; break;
            }
            if (value < 0 || amount < 0 || amount > 1 ||
                minLength < 0) { SetError(error, kBadValue); valid = false; break; }
            const float t = HairT(hairT.data, i, b, e, error);
            // usdGen:mask IS the operator envelope (02 §2.13).
            const float envelope = amount;
            if (!isfinite(envelope) || envelope < 0 || envelope > 1) {
                SetError(error, kBadValue); valid = false; break;
            }
            // Exact zero envelope must not round-trip points through subtract/
            // multiply/add, and must not require a direction for a zero curve.
            if (envelope == 0.0f) { output.data[i] = g.points.data[i]; continue; }
            const float localTarget = fmaxf(p.mode == LengthMode::Set
                ? value * multiplier : current * value * multiplier, minLength);
            if (!isfinite(localTarget)) { SetError(error, kNonFinite); valid = false; break; }
            if (current == 0.0f && localTarget > 0.0f) {
                // No tangent exists from which to infer an extension direction.
                SetError(error, kBadValue); valid = false; break;
            }
            const float factor = current > 0.0f ? localTarget / current : 1.0f;
            if (!isfinite(factor) || factor < 0) {
                SetError(error, kBadValue); valid = false; break;
            }
            const float3 input = g.points.data[i];
            float3 changed = input;
            if (p.method == LengthMethod::Scale) {
                changed = make_float3(root.x+(input.x-root.x)*factor,
                                      root.y+(input.y-root.y)*factor,
                                      root.z+(input.z-root.z)*factor);
            } else {
                float distance = 0;
                if (p.rebuild == LengthRebuild::Reparam) distance = t * localTarget;
                else {
                    for (uint32_t j = b + 1u; j <= i; ++j) {
                        const float3 a = g.points.data[j-1u], q = g.points.data[j];
                        const float3 d = make_float3(q.x-a.x, q.y-a.y, q.z-a.z);
                        distance += sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
                    }
                    distance = fminf(distance, localTarget);
                }
                // With keepParam, original CV positions are retained while
                // extending the final tip to the requested arc length.
                if (localTarget > current && i + 1u == e)
                    distance = localTarget;
                if (localTarget > current && distance > current && e > b + 1u) {
                    uint32_t j = e - 1u;
                    bool found = false;
                    float3 direction = make_float3(0,0,0);
                    float directionNorm = 0.0f;
                    while (j > b) {
                        const float3 a = g.points.data[j-1u], q = g.points.data[j];
                        direction = make_float3(q.x-a.x, q.y-a.y, q.z-a.z);
                        directionNorm = sqrtf(direction.x*direction.x +
                                              direction.y*direction.y +
                                              direction.z*direction.z);
                        if (directionNorm > 1.0e-12f) { found = true; break; }
                        --j;
                    }
                    if (found) {
                        const float3 tip = g.points.data[e-1u];
                        const float extra = distance - current;
                        changed = make_float3(tip.x+direction.x/directionNorm*extra,
                                              tip.y+direction.y/directionNorm*extra,
                                              tip.z+direction.z/directionNorm*extra);
                    } else changed = g.points.data[e-1u];
                } else changed = SampleArc(g.points.data, b, e, distance);
            }
            const float3 result = make_float3(input.x+(changed.x-input.x)*envelope,
                                              input.y+(changed.y-input.y)*envelope,
                                              input.z+(changed.z-input.z)*envelope);
            if (!Finite(result)) { SetError(error, kNonFinite); valid = false; break; }
            output.data[i] = result;
            if (envelope > 0.0f) allZeroEnvelope = false;
        }
        if (!valid) continue;
        float resulting = 0;
        for (uint32_t i = b + 1u; i < e; ++i) {
            const float3 a = output.data[i-1u], q = output.data[i];
            const float3 d = make_float3(q.x-a.x, q.y-a.y, q.z-a.z);
            resulting += sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
        }
        if (!isfinite(resulting)) { SetError(error, kNonFinite); continue; }
        keep.data[c] = allZeroEnvelope || resulting >= threshold ? 1 : 0;
    }
}

// Fresh publication is conditional: a device semantic rejection must leave
// both caller-owned candidate channels untouched.
__global__ void PublishLength(DeviceView<const float3> staging,
                              DeviceView<const uint8_t> keepStaging,
                              DeviceView<float3> output,
                              DeviceView<uint8_t> keep,
                              const int* error) {
    if (LoadError(error) != 0) return;
    const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(gridDim.x) * blockDim.x;
    for (size_t i = first; i < staging.size; i += stride) output.data[i] = staging.data[i];
    for (size_t i = first; i < keepStaging.size; i += stride) keep.data[i] = keepStaging.data[i];
}

unsigned Blocks(size_t count) {
    const size_t required = (count + 255u) / 256u;
    return static_cast<unsigned>(std::max<size_t>(1u,
        std::min<size_t>(65535u, required)));
}
StyleStatus Decode(int code) {
    return code == kBadOffsets ? StyleStatus::InvalidArgument :
           code == kNonFinite ? StyleStatus::NonFiniteInput :
           code == kBadValue ? StyleStatus::InvalidValue : StyleStatus::CudaError;
}
} // namespace

CudaLength::~CudaLength() {
    if (unprovenWork_) {
        error_.quarantine(); staging_.quarantine(); keepStaging_.quarantine();
        ready_ = nullptr; freshHostError_ = nullptr; freshHostErrorPermit_.Abandon();
        return;
    }
    const bool owns = error_.size() || staging_.size() || keepStaging_.size() ||
        ready_ || freshHostError_;
    if (!owns) return;
    int previous = -1;
    const bool gotPrevious = cudaGetDevice(&previous) == cudaSuccess;
    const bool selected = deviceIndex_ >= 0 && cudaSetDevice(deviceIndex_) == cudaSuccess;
    // shutdown wait: destructor.
    const bool synchronized = selected && (!ready_ || cudaEventSynchronize(ready_) == cudaSuccess);
    if (!selected || !synchronized) {
        error_.quarantine(); staging_.quarantine(); keepStaging_.quarantine();
        ready_ = nullptr; freshHostError_ = nullptr; freshHostErrorPermit_.Abandon();
        if (selected && gotPrevious && previous != deviceIndex_) cudaSetDevice(previous);
        return;
    }
    if (ready_) cudaEventDestroy(ready_);
    ready_ = nullptr;
    error_.release(); staging_.release(); keepStaging_.release();
    if (freshHostError_) {
        if (selected && cudaFreeHost(freshHostError_) == cudaSuccess)
            freshHostErrorPermit_.Release();
        else
            freshHostErrorPermit_.Abandon();
        freshHostError_ = nullptr;
    }
    if (gotPrevious && previous != deviceIndex_) cudaSetDevice(previous);
}

StyleStatus CudaLength::fail(StyleStatus status, const char *message) {
    diagnostic_ = message; return status;
}

StyleStatus CudaLength::validateField(ScalarField f,
                                      DeviceCurveGeometryView g) const {
    const size_t expected = f.domain == expr::Domain::Groom ? 1u :
                            f.domain == expr::Domain::Primitive ? g.curveCount :
                            f.domain == expr::Domain::Point ? g.pointCount :
                            std::numeric_limits<size_t>::max();
    if (expected == std::numeric_limits<size_t>::max()) return StyleStatus::InvalidArgument;
    if (!f.data && f.count == 0) {
        if (expected == 0) return StyleStatus::Ok;
        if (f.domain != expr::Domain::Groom) return StyleStatus::InvalidArgument;
        return std::isfinite(f.literal) ? StyleStatus::Ok : StyleStatus::NonFiniteInput;
    }
    return f.data && f.count == expected ? StyleStatus::Ok : StyleStatus::InvalidArgument;
}

StyleStatus CudaLength::validateVec2(Vec2Field f,
                                     DeviceCurveGeometryView g) const {
    const size_t expected = f.domain == expr::Domain::Groom ? 1u :
                            f.domain == expr::Domain::Primitive ? g.curveCount :
                            f.domain == expr::Domain::Point ? g.pointCount :
                            std::numeric_limits<size_t>::max();
    if (expected == std::numeric_limits<size_t>::max()) return StyleStatus::InvalidArgument;
    if (!f.data && f.count == 0) {
        if (expected == 0) return StyleStatus::Ok;
        if (f.domain != expr::Domain::Groom) return StyleStatus::InvalidArgument;
        return std::isfinite(f.literal.x) && std::isfinite(f.literal.y)
                   ? StyleStatus::Ok : StyleStatus::NonFiniteInput;
    }
    return f.data && f.count == expected ? StyleStatus::Ok : StyleStatus::InvalidArgument;
}

StyleStatus CudaLength::validateBool(BoolField f,
                                      DeviceCurveGeometryView g) const {
    const size_t expected = f.domain == expr::Domain::Groom ? 1u :
                            f.domain == expr::Domain::Primitive ? g.curveCount :
                            std::numeric_limits<size_t>::max();
    if (expected == std::numeric_limits<size_t>::max()) return StyleStatus::InvalidArgument;
    if (!f.data && f.count == 0) {
        if (expected == 0) return StyleStatus::Ok;
        return f.domain == expr::Domain::Groom ? StyleStatus::Ok : StyleStatus::InvalidArgument;
    }
    return f.data && f.count == expected ? StyleStatus::Ok : StyleStatus::InvalidArgument;
}

StyleStatus CudaLength::Apply(DeviceCurveGeometryView g, DeviceView<const float> hairT,
                              LengthParameters p, DeviceView<float3> output,
                              DeviceView<uint8_t> keep, cudaStream_t stream,
                              UsdGenExecutionMemoryReservation* reservation) {
    if (pending_ || (freshPreparing_ && !freshApplying_) || freshPending_ ||
        unprovenWork_ || freshUploadFailed_) return fail(StyleStatus::InvalidArgument,
                              "Finish is required before another Length operation");
    if (g.curveCount > size_t(INT_MAX) || g.pointCount > size_t(UINT32_MAX) ||
        g.curveCount == std::numeric_limits<size_t>::max() ||
        (g.curveCount == 0 && g.pointCount != 0) || g.points.size != g.pointCount ||
        g.curveOffsets.size != g.curveCount + 1u || !g.curveOffsets.data ||
        (g.pointCount && !g.points.data) || output.size != g.pointCount ||
        keep.size != g.curveCount || (g.pointCount && !output.data) ||
        (g.curveCount && !keep.data) ||
        (g.stableIds.data && g.stableIds.size != g.curveCount) ||
        (!g.stableIds.data && g.stableIds.size != 0) ||
        (!hairT.data && hairT.size != 0) ||
        (hairT.data && hairT.size != g.pointCount))
        return fail(StyleStatus::InvalidArgument, "invalid Length geometry or views");
    const ScalarField fields[] = {p.value, p.mask,
                                  p.minRemainingLength, p.cullThreshold};
    for (ScalarField f : fields) {
        const StyleStatus s = validateField(f, g);
        if (s != StyleStatus::Ok) return fail(s, "invalid Length scalar field");
    }
    StyleStatus s = validateVec2(p.random, g);
    if (s != StyleStatus::Ok) return fail(s, "invalid Length random field");
    s = validateBool(p.enabled, g);
    if (s != StyleStatus::Ok) return fail(s, "invalid Length enabled field");
    if (p.random.domain == expr::Domain::Point)
        return fail(StyleStatus::InvalidArgument, "Length random is not point-domain");
    if (p.cullThreshold.domain == expr::Domain::Point)
        return fail(StyleStatus::InvalidArgument, "Length cullThreshold is not point-domain");
    if (p.mode != LengthMode::Set && p.mode != LengthMode::Scale && p.mode != LengthMode::Cull)
        return fail(StyleStatus::InvalidArgument, "unknown Length mode");
    if (p.method != LengthMethod::Scale && p.method != LengthMethod::CutExtend)
        return fail(StyleStatus::InvalidArgument, "unknown Length method");
    if (p.rebuild != LengthRebuild::KeepParam && p.rebuild != LengthRebuild::Reparam)
        return fail(StyleStatus::InvalidArgument, "unknown Length rebuild mode");

    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess)
        return fail(StyleStatus::CudaError, "cannot identify CUDA device");
    if (deviceIndex_ >= 0 && current != deviceIndex_)
        return fail(StyleStatus::InvalidArgument, "Length device mismatch");
    if (stream) {
        int streamDevice = -1;
        if (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess)
            return fail(StyleStatus::CudaError, "cannot identify Length stream");
        if (streamDevice != current)
            return fail(StyleStatus::InvalidArgument, "Length stream device mismatch");
    }
    if (deviceIndex_ < 0) deviceIndex_ = current;
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return fail(StyleStatus::CudaError, "Length event creation failed");
    if (error_.reset(1, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        staging_.reset(g.pointCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess ||
        keepStaging_.reset(g.curveCount, reservation, UsdGenExecutionResourceKind::Scratch) != cudaSuccess)
        return fail(StyleStatus::CudaError, "Length allocation failed");
    if (freshPreparing_ && !freshHostError_) {
        auto permit = TryReserveCudaExecutionBytes(sizeof(int), UsdGenExecutionResourceKind::Scratch, reservation);
        int* hostError = nullptr;
        if (!permit || cudaHostAlloc(reinterpret_cast<void**>(&hostError), sizeof(int), cudaHostAllocDefault) != cudaSuccess)
            return fail(StyleStatus::CudaError, "Length pinned diagnostic allocation failed");
        freshHostError_ = hostError;
        freshHostErrorPermit_ = std::move(*permit);
    }
    if (freshPreparing_) unprovenWork_ = true;
    if (cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess) {
        // A failed enqueue cannot prove that an earlier operation on this
        // stream has completed.  The destructor must quarantine this state.
        unprovenWork_ = true;
        return fail(StyleStatus::CudaError, "Length error reset failed");
    }
    ValidateKernel<<<Blocks(std::max(g.curveCount, g.pointCount)), 256, 0, stream>>>(
        g, hairT, error_.data());
    if (cudaGetLastError() != cudaSuccess) {
        unprovenWork_ = true;
        return fail(StyleStatus::CudaError, "Length validation launch failed");
    }
    LengthKernel<<<Blocks(g.curveCount), 256, 0, stream>>>(
        g, hairT, p, staging_.view(), keepStaging_.view(), error_.data());
    if (cudaGetLastError() != cudaSuccess) {
        unprovenWork_ = true;
        return fail(StyleStatus::CudaError, "Length launch failed");
    }
    if (cudaEventRecord(ready_, stream) != cudaSuccess) {
        unprovenWork_ = true;
        return fail(StyleStatus::CudaError, "Length event record failed");
    }
    output_ = output; keep_ = keep; points_ = g.pointCount; curves_ = g.curveCount;
    pending_ = true;
    return StyleStatus::Ok;
}

StyleStatus CudaLength::Finish(cudaStream_t stream) {
    if (!pending_ || freshPending_ || unprovenWork_) return StyleStatus::InvalidArgument;
    int current = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != deviceIndex_) {
        unprovenWork_ = true;
        return StyleStatus::InvalidArgument;
    }
    if (stream) {
        int streamDevice = -1;
        if (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess) {
            unprovenWork_ = true;
            return StyleStatus::CudaError;
        }
        if (streamDevice != deviceIndex_) {
            unprovenWork_ = true;
            return StyleStatus::InvalidArgument;
        }
    }
    int code = 0;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(&code, error_.data(), sizeof(code), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess) {
        unprovenWork_ = true;
        pending_ = false;
        return fail(StyleStatus::CudaError, "Length diagnostic readback failed");
    }
    if (code) {
        pending_ = false; output_ = {}; keep_ = {}; points_ = curves_ = 0;
        return fail(Decode(code), "Length device validation failed");
    }
    if ((points_ && cudaMemcpyAsync(output_.data, staging_.data(), points_*sizeof(float3),
                                    cudaMemcpyDeviceToDevice, stream) != cudaSuccess) ||
        (curves_ && cudaMemcpyAsync(keep_.data, keepStaging_.data(), curves_*sizeof(uint8_t),
                                     cudaMemcpyDeviceToDevice, stream) != cudaSuccess) ||
        CudaFenceStream(stream) != cudaSuccess) {
        unprovenWork_ = true;
        pending_ = false;
        return fail(StyleStatus::CudaError, "Length publication failed");
    }
    pending_ = false; output_ = {}; keep_ = {}; points_ = curves_ = 0;
    return StyleStatus::Ok;
}

StyleStatus CudaLength::ApplyFresh(DeviceCurveGeometryView geometry,
                                   DeviceView<const float> hairT,
                                   LengthParameters parameters,
                                   DeviceView<float3> output,
                                   DeviceView<uint8_t> keep,
                                   cudaStream_t stream,
                                   UsdGenExecutionMemoryReservation* reservation) {
    if (pending_ || freshPending_ || freshPreparing_ || unprovenWork_)
        return StyleStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return StyleStatus::InvalidArgument;
    freshPreparing_ = true;
    freshApplying_ = true;
    StyleStatus const status = Apply(geometry, hairT, parameters, output, keep,
                                     stream, reservation);
    freshApplying_ = false;
    freshPreparing_ = false;
    if (status != StyleStatus::Ok) {
        if (unprovenWork_) freshUploadFailed_ = true;
        return status;
    }
    freshPending_ = true;
    freshUploadFailed_ = false;
    freshCallbackArmed_ = false;
    *freshHostError_ = std::numeric_limits<int>::min();
    return StyleStatus::Ok;
}

StyleStatus CudaLength::FinishFreshAsync(cudaStream_t stream,
    void (*callback)(cudaStream_t, cudaError_t, void*) noexcept, void* userdata) {
    if (!pending_ || !freshPending_ || !callback || freshUploadFailed_ ||
        freshCallbackArmed_ || !freshHostError_)
        return StyleStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) {
        freshUploadFailed_ = true;
        return StyleStatus::InvalidArgument;
    }
    int current = -1, streamDevice = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != deviceIndex_ ||
        (stream && (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess ||
                    streamDevice != deviceIndex_)) ||
        cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess) {
        freshUploadFailed_ = true;
        return StyleStatus::CudaError;
    }
    if (points_ || curves_) {
        PublishLength<<<Blocks(std::max(points_, curves_)), 256, 0, stream>>>(
            {staging_.data(), staging_.size()}, {keepStaging_.data(), keepStaging_.size()},
            output_, keep_, error_.data());
        if (cudaGetLastError() != cudaSuccess) {
            freshUploadFailed_ = true;
            return StyleStatus::CudaError;
        }
    }
    if (cudaMemcpyAsync(freshHostError_, error_.data(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamAddCallback(stream, callback, userdata, 0) != cudaSuccess) {
        freshUploadFailed_ = true;
        return StyleStatus::CudaError;
    }
    freshCallbackArmed_ = true;
    return StyleStatus::Ok;
}

StyleStatus CudaLength::CommitFreshFinish() {
    if (!pending_ || !freshPending_ || !freshCallbackArmed_ || freshUploadFailed_ ||
        !freshHostError_ || *freshHostError_ == std::numeric_limits<int>::min())
        return StyleStatus::InvalidArgument;
    pending_ = false; freshPending_ = false; freshCallbackArmed_ = false;
    freshUploadFailed_ = false; unprovenWork_ = false;
    points_ = curves_ = 0; output_ = {}; keep_ = {};
    return *freshHostError_ == 0 ? StyleStatus::Ok : Decode(*freshHostError_);
}
} // namespace usdGen::gpu
