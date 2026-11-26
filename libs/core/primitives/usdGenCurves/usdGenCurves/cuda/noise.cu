#include "usdGenCurves/cuda/noise.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenCuda/gpu/cudaCompat.h"

#include "usdGenExpr/cuda/seexprNoise.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace usdGen::gpu {
namespace {

constexpr size_t kProfileSize = 257;
constexpr int kBadOffsets = 1;
constexpr int kNonFinite = 2;
constexpr int kBadValue = 3;
constexpr uint32_t kSaltNoise = 0x4E01523Eu;

__device__ void SetError(int* error, int code) { atomicCAS(error, 0, code); }
__device__ int LoadError(int const* error) {
    return atomicAdd(const_cast<int*>(error), 0);
}
__device__ bool Finite(float value) { return isfinite(value); }
__device__ bool Finite(float3 value) {
    return Finite(value.x) && Finite(value.y) && Finite(value.z);
}
__device__ float Dot(float3 a, float3 b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}
__device__ float3 Cross(float3 a, float3 b) {
    return make_float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
__device__ float Length(float3 v) { return sqrtf(Dot(v, v)); }
__device__ float3 Add(float3 a, float3 b) {
    return make_float3(a.x+b.x, a.y+b.y, a.z+b.z);
}
__device__ float3 Sub(float3 a, float3 b) {
    return make_float3(a.x-b.x, a.y-b.y, a.z-b.z);
}
__device__ float3 Mul(float3 a, float b) {
    return make_float3(a.x*b, a.y*b, a.z*b);
}

__device__ size_t FindFrame(DeviceView<const uint64_t> frameIds,
                            uint64_t id) {
    size_t lo = 0, hi = frameIds.size;
    while (lo < hi) {
        size_t const mid = lo + (hi - lo) / 2u;
        uint64_t const value = frameIds.data[mid];
        if (value < id) lo = mid + 1u;
        else hi = mid;
    }
    return lo < frameIds.size && frameIds.data[lo] == id ? lo : frameIds.size;
}

// Exact device spelling of usdGenMath's SplitMix64 draw, retained locally
// because the host header is not annotated for device compilation.
__device__ uint64_t Hash64(uint64_t key, uint32_t salt) {
    uint64_t z = (key ^ (uint64_t(salt) * 0x9E3779B97F4A7C15ull)) +
                 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}
__device__ float Draw01(int seed, uint64_t curveId, uint32_t salt) {
    const uint64_t key = Hash64(uint64_t(uint32_t(seed)), salt) ^ curveId;
    return float(uint32_t(Hash64(key, salt) >> 32) >> 8) * 0x1.0p-24f;
}

// Mirrors SeExpr2::FBM<3,3,false>: each vector component advances all three
// input coordinates by +1000 before evaluating the next component at every
// octave.  Do not replace this with three independent hashes/noise calls.
// Noise3 floors into an int lattice and samples its +1 neighbor.  Check the
// float value in double first: float(INT_MAX) is already 2^31, so a float
// comparison against INT_MAX would accidentally admit an overflowing cast.
__device__ bool SupportedLatticeCoordinate(float value) {
    if (!Finite(value)) return false;
    const double cell = floor(static_cast<double>(value));
    return cell >= -2147483648.0 && cell < 2147483647.0;
}

__device__ bool SupportedNoisePoint(float x, float y, float z) {
    return SupportedLatticeCoordinate(x) && SupportedLatticeCoordinate(y) &&
        SupportedLatticeCoordinate(z);
}

__device__ bool Vfbm3(float x, float y, float z, int octaves,
                      float lacunarity, float gain, float3* result) {
    *result = make_float3(0, 0, 0);
    float scale = 1.0f;
    for (int octave = 0; octave < octaves; ++octave) {
        const float x1 = x + 1000.0f, y1 = y + 1000.0f, z1 = z + 1000.0f;
        const float x2 = x + 2000.0f, y2 = y + 2000.0f, z2 = z + 2000.0f;
        if (!SupportedNoisePoint(x, y, z) || !SupportedNoisePoint(x1, y1, z1) ||
            !SupportedNoisePoint(x2, y2, z2)) return false;
        result->x += usdgen_noise::Noise3(x, y, z) * scale;
        result->y += usdgen_noise::Noise3(x1, y1, z1) * scale;
        result->z += usdgen_noise::Noise3(x2, y2, z2) * scale;
        if (octave + 1 == octaves) break;
        scale *= gain;
        x = x * lacunarity + 1234.0f;
        y = y * lacunarity + 1234.0f;
        z = z * lacunarity + 1234.0f;
        if (!Finite(x) || !Finite(y) || !Finite(z) || !Finite(scale)) return false;
    }
    return Finite(*result);
}

__device__ bool ReadScalar(ScalarField field, size_t curve, size_t point,
                           float* value, int* error) {
    size_t index = 0;
    if (!field.data) {
        if (field.count != 0 || field.domain != expr::Domain::Groom) {
            SetError(error, kBadValue); return false;
        }
        *value = field.literal;
    } else {
        if (field.domain == expr::Domain::Primitive) index = curve;
        else if (field.domain == expr::Domain::Point) index = point;
        else if (field.domain != expr::Domain::Groom) {
            SetError(error, kBadValue); return false;
        }
        if (index >= field.count) { SetError(error, kBadValue); return false; }
        *value = field.data[index];
    }
    if (!Finite(*value)) { SetError(error, kNonFinite); return false; }
    return true;
}

__device__ bool ReadBool(BoolField field, size_t curve, bool* value,
                         int* error) {
    size_t index = 0;
    uint8_t raw = field.literal ? 1u : 0u;
    if (!field.data) {
        if (field.count != 0 || field.domain != expr::Domain::Groom) {
            SetError(error, kBadValue); return false;
        }
    } else {
        if (field.domain == expr::Domain::Primitive) index = curve;
        else if (field.domain != expr::Domain::Groom) {
            SetError(error, kBadValue); return false;
        }
        if (index >= field.count) { SetError(error, kBadValue); return false; }
        raw = field.data[index];
    }
    if (raw > 1u) { SetError(error, kBadValue); return false; }
    *value = raw != 0u;
    return true;
}

__device__ bool ReadInt(IntField field, size_t curve, size_t point,
                        int32_t* value, int* error) {
    size_t index = 0;
    if (!field.data) {
        if (field.count != 0 || field.domain != expr::Domain::Groom) {
            SetError(error, kBadValue); return false;
        }
        *value = field.literal;
    } else {
        if (field.domain == expr::Domain::Primitive) index = curve;
        else if (field.domain == expr::Domain::Point) index = point;
        else if (field.domain != expr::Domain::Groom) {
            SetError(error, kBadValue); return false;
        }
        if (index >= field.count) { SetError(error, kBadValue); return false; }
        *value = field.data[index];
    }
    if (*value < 1 || *value > 6) { SetError(error, kBadValue); return false; }
    return true;
}

__device__ bool ReadSeed(IntField field, size_t curve, int32_t* value,
                         int* error) {
    if (!field.data) {
        if (field.domain != expr::Domain::Groom || field.count != 0) {
            SetError(error, kBadValue); return false;
        }
        *value = field.literal;
        return true;
    }
    size_t index = 0;
    if (field.domain == expr::Domain::Primitive) index = curve;
    else if (field.domain != expr::Domain::Groom) {
        SetError(error, kBadValue); return false;
    }
    if (index >= field.count) { SetError(error, kBadValue); return false; }
    *value = field.data[index];
    return true;
}

__device__ float HairT(DeviceView<const float> hairT, uint32_t point,
                       uint32_t begin, uint32_t end, int* error) {
    float result = end > begin + 1u
        ? float(point - begin) / float(end - begin - 1u) : 0.0f;
    if (hairT.data) result = hairT.data[point];
    if (!Finite(result) || result < 0.0f || result > 1.0f) {
        SetError(error, Finite(result) ? kBadValue : kNonFinite);
        return 0.0f;
    }
    return result;
}

__device__ float Sample257(DeviceView<const float> profile, float t) {
    if (!profile.data) return 1.0f;
    const float coordinate = t * 256.0f;
    const int lower = min(255, max(0, int(floorf(coordinate))));
    const float fraction = coordinate - float(lower);
    return profile.data[lower] + (profile.data[lower + 1] - profile.data[lower]) * fraction;
}

__global__ void ValidateInputs(DeviceCurveGeometryView geometry,
                               DeviceView<const float> hairT,
                               RestRootFrames frames,
                               DeviceView<const float> magnitudeProfile,
                               int* error) {
    const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    if (geometry.curveCount) {
        for (size_t c = first; c < geometry.curveCount; c += stride) {
            const uint32_t begin = geometry.curveOffsets.data[c];
            const uint32_t end = geometry.curveOffsets.data[c + 1u];
            // Test every pair, including the final one.  Checking only the
            // terminal offset accidentally accepted [0, 3, 3], then let the
            // kernel dereference restPoints[3] for the empty final curve.
            if ((c == 0 && begin != 0u) || begin >= end ||
                end > geometry.pointCount ||
                (c + 1u == geometry.curveCount && end != geometry.pointCount))
                SetError(error, kBadOffsets);
            size_t frameCurve = c;
            if (frames.stableIds.data) {
                frameCurve = FindFrame(frames.stableIds, geometry.stableIds.data[c]);
                if (frameCurve == frames.stableIds.size) {
                    SetError(error, kBadValue);
                    continue;
                }
            }
            const float3 tangent = frames.tangent.data[frameCurve];
            const float3 binormal = frames.binormal.data[frameCurve];
            const float3 normal = frames.normal.data[frameCurve];
            const float tangentLength = Length(tangent);
            const float binormalLength = Length(binormal);
            const float normalLength = Length(normal);
            if (!Finite(tangent) || !Finite(normal) || !Finite(binormal))
                SetError(error, kNonFinite);
            else if (fabsf(tangentLength - 1.0f) > 2.0e-4f ||
                     fabsf(binormalLength - 1.0f) > 2.0e-4f ||
                     fabsf(normalLength - 1.0f) > 2.0e-4f ||
                     fabsf(Dot(tangent, binormal)) > 2.0e-4f ||
                     fabsf(Dot(tangent, normal)) > 2.0e-4f ||
                     fabsf(Dot(binormal, normal)) > 2.0e-4f ||
                     Dot(Cross(tangent, binormal), normal) < 0.999f)
                SetError(error, kBadValue);
        }
    } else if (geometry.curveOffsets.size == 1u) {
        // The canonical empty C3 topology owns one terminal offset.  Keep
        // this check on the device: callers may only provide a device view,
        // so validating the value on the host would require an otherwise
        // forbidden synchronous readback.  A non-zero offset is malformed
        // even though it cannot be consumed by NoiseKernel (which has no
        // curve launch for an empty topology).
        if (!geometry.curveOffsets.data || geometry.curveOffsets.data[0] != 0u)
            SetError(error, kBadOffsets);
    }
    for (size_t point = first; point < geometry.pointCount; point += stride) {
        if (!Finite(geometry.points.data[point]) || !Finite(geometry.restPoints.data[point]))
            SetError(error, kNonFinite);
        if (hairT.data && (!Finite(hairT.data[point]) || hairT.data[point] < 0.0f ||
                           hairT.data[point] > 1.0f))
            SetError(error, Finite(hairT.data[point]) ? kBadValue : kNonFinite);
    }
    for (size_t i = first; i < magnitudeProfile.size; i += stride) {
        const float value = magnitudeProfile.data[i];
        if (!Finite(value) || value < 0.0f)
            SetError(error, Finite(value) ? kBadValue : kNonFinite);
    }
}

__global__ void NoiseKernel(DeviceCurveGeometryView geometry,
                            DeviceView<const float> hairT,
                            RestRootFrames frames, NoiseParameters parameters,
                            float3* output, int* error) {
    const size_t first = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = size_t(blockDim.x) * gridDim.x;
    for (size_t curve = first; curve < geometry.curveCount; curve += stride) {
        if (LoadError(error) != 0) continue;
        const uint32_t begin = geometry.curveOffsets.data[curve];
        const uint32_t end = geometry.curveOffsets.data[curve + 1u];
        bool enabled = true, cumulative = false;
        int32_t seed = 0;
        if (!ReadBool(parameters.enabled, curve, &enabled, error) ||
            !ReadBool(parameters.cumulative, curve, &cumulative, error) ||
            !ReadSeed(parameters.seed, curve, &seed, error)) continue;
        size_t frameCurve = curve;
        if (frames.stableIds.data) {
            frameCurve = FindFrame(frames.stableIds, geometry.stableIds.data[curve]);
            if (frameCurve == frames.stableIds.size) {
                SetError(error, kBadValue); continue;
            }
        }
        const float3 tangent = frames.tangent.data[frameCurve];
        const float3 binormal = frames.binormal.data[frameCurve];
        const float3 normal = frames.normal.data[frameCurve];
        const float3 rootRest = geometry.restPoints.data[begin];
        const uint64_t stableId = geometry.stableIds.data[curve];
        const float3 hash = make_float3(
            Draw01(seed, stableId, kSaltNoise),
            Draw01(seed, stableId, kSaltNoise + 1u),
            Draw01(seed, stableId, kSaltNoise + 2u));
        float3 running = make_float3(0, 0, 0);
        float3 previousOutput = geometry.points.data[begin];
        for (uint32_t point = begin; point < end; ++point) {
            float magnitude = 0.0f, frequency = 0.0f, correlation = 0.0f;
            float lacunarity = 0.0f, gain = 0.0f, preserve = 0.0f;
            float mask = 0.0f;
            int32_t octaves = 0;
            if (!ReadScalar(parameters.magnitude, curve, point, &magnitude, error) ||
                !ReadScalar(parameters.frequency, curve, point, &frequency, error) ||
                !ReadScalar(parameters.correlation, curve, point, &correlation, error) ||
                !ReadInt(parameters.octaves, curve, point, &octaves, error) ||
                !ReadScalar(parameters.lacunarity, curve, point, &lacunarity, error) ||
                !ReadScalar(parameters.gain, curve, point, &gain, error) ||
                !ReadScalar(parameters.preserveLength, curve, point, &preserve, error) ||
                !ReadScalar(parameters.mask, curve, point, &mask, error))
                continue;
            if (magnitude < 0.0f || frequency <= 0.0f || correlation < 0.0f ||
                correlation > 1.0f || lacunarity <= 1.0f || gain < 0.0f || gain > 1.0f ||
                preserve < 0.0f || preserve > 1.0f ||
                mask < 0.0f || mask > 1.0f) {
                SetError(error, kBadValue); continue;
            }
            const float t = HairT(hairT, point, begin, end, error);
            const float magnitudeRamp = Sample257(parameters.magnitudeProfile, t);
            // usdGen:mask IS the operator envelope (02 §2.13).
            const float envelope = mask;
            if (!Finite(magnitudeRamp) || envelope < 0.0f || envelope > 1.0f) {
                SetError(error, !Finite(magnitudeRamp) ? kNonFinite : kBadValue);
                continue;
            }
            const float3 input = geometry.points.data[point];
            if (!enabled) {
                output[point] = input;
                previousOutput = input;
                continue;
            }
            const float baseX = rootRest.x * correlation + hash.x * (1.0f - correlation);
            const float baseY = rootRest.y * correlation + hash.y * (1.0f - correlation);
            const float baseZ = rootRest.z * correlation + hash.z * (1.0f - correlation) +
                                t * frequency;
            float3 field{};
            if (!Vfbm3(baseX, baseY, baseZ, octaves, lacunarity, gain, &field)) {
                SetError(error, kBadValue); continue;
            }
            // §2.9 accumulates the vector field, then applies this CV's
            // magnitude/ramp/envelope.  In particular, a zero root ramp must
            // not suppress field accumulation used by a later CV.
            float3 accumulatedField = field;
            if (cumulative) {
                running = Add(running, field);
                accumulatedField = running;
            }
            // §0.5 requires a bitwise input write for a zero resolved
            // envelope.  Keep this after accumulation so a zero-ramp/root
            // CV still contributes field state for a later cumulative CV,
            // but before displacement/restoration so mask==0
            // cannot alter an already-styled segment (including signed zero).
            if (magnitude == 0.0f || magnitudeRamp == 0.0f || envelope == 0.0f) {
                output[point] = input;
                previousOutput = input;
                continue;
            }
            const float3 displacement = Mul(accumulatedField,
                magnitude * magnitudeRamp * envelope);
            const float3 worldDisplacement = Add(Add(Mul(tangent, displacement.x),
                                                       Mul(binormal, displacement.y)),
                                                Mul(normal, displacement.z));
            const float3 displaced = Add(input, worldDisplacement);
            if (!Finite(displaced)) { SetError(error, kNonFinite); continue; }
            float3 restored = displaced;
            if (point > begin && preserve != 0.0f) {
                const float restLength = Length(Sub(geometry.restPoints.data[point],
                                                    geometry.restPoints.data[point - 1u]));
                const float displacedLength = Length(Sub(displaced, previousOutput));
                if (!Finite(restLength) || !Finite(displacedLength)) {
                    SetError(error, kNonFinite); continue;
                }
                if (restLength == 0.0f) restored = previousOutput;
                else if (displacedLength == 0.0f) {
                    SetError(error, kBadValue); continue;
                } else {
                    restored = Add(previousOutput,
                        Mul(Sub(displaced, previousOutput), restLength / displacedLength));
                }
            }
            const float3 result = preserve == 0.0f ? displaced :
                Add(Mul(displaced, 1.0f - preserve), Mul(restored, preserve));
            if (!Finite(result)) { SetError(error, kNonFinite); continue; }
            output[point] = result;
            previousOutput = result;
        }
    }
}

unsigned Blocks(size_t count) {
    return static_cast<unsigned>(std::max<size_t>(
        1, std::min<size_t>((count + 255u) / 256u, 65535u)));
}

StyleStatus DecodeError(int code) {
    if (code == kBadOffsets) return StyleStatus::InvalidArgument;
    if (code == kNonFinite) return StyleStatus::NonFiniteInput;
    if (code == kBadValue) return StyleStatus::InvalidValue;
    return StyleStatus::CudaError;
}

} // namespace

CudaNoise::~CudaNoise() {
    if (unprovenWork_ || freshCallbackArmed_) {
        staging_.quarantine();
        error_.quarantine();
        ready_ = nullptr;
        freshHostError_ = nullptr;
        freshHostErrorPermit_.Abandon();
        return;
    }
    const bool ownsResources = ready_ || staging_.size() || error_.size() ||
        freshHostError_;
    int previous = -1;
    const bool selected = !ownsResources || (deviceIndex_ >= 0 &&
        cudaGetDevice(&previous) == cudaSuccess &&
        cudaSetDevice(deviceIndex_) == cudaSuccess);
    const bool complete = !ownsResources ||
        (selected && (!pending_ || !ready_ || cudaEventSynchronize(ready_) == cudaSuccess));
    if (!complete) {
        staging_.quarantine();
        error_.quarantine();
        ready_ = nullptr;
        // The status D2H may still target this allocation.  Losing the
        // completion proof means it is intentionally leaked together with
        // its charge; releasing the permit would make the pool claim bytes
        // are reusable while CUDA may still own them.
        freshHostError_ = nullptr;
        freshHostErrorPermit_.Abandon();
    } else {
        if (ready_) cudaEventDestroy(ready_);
        staging_.reset(0);
        error_.reset(0);
        if (freshHostError_) {
            if (cudaFreeHost(freshHostError_) == cudaSuccess)
                freshHostErrorPermit_.Release();
            else
                freshHostErrorPermit_.Abandon();
            freshHostError_ = nullptr;
        }
    }
    if (selected && previous >= 0 && previous != deviceIndex_) cudaSetDevice(previous);
}

StyleStatus CudaNoise::validateScalar(ScalarField field,
                                      DeviceCurveGeometryView geometry) const {
    if (field.domain != expr::Domain::Groom && field.domain != expr::Domain::Primitive &&
        field.domain != expr::Domain::Point) return StyleStatus::InvalidArgument;
    if (!field.data) {
        if (field.count != 0) return StyleStatus::InvalidArgument;
        // Empty parameter evaluation deliberately preserves its declared
        // Point/Primitive domain as {nullptr, 0, domain}; it is not a groom
        // literal and no thread will read it.  A nonempty geometry cannot
        // silently reinterpret that missing field as a literal.
        if (field.domain != expr::Domain::Groom) {
            const size_t expected = field.domain == expr::Domain::Primitive
                ? geometry.curveCount : geometry.pointCount;
            return expected == 0 ? StyleStatus::Ok : StyleStatus::InvalidArgument;
        }
        return std::isfinite(field.literal) ? StyleStatus::Ok : StyleStatus::NonFiniteInput;
    }
    const size_t expected = field.domain == expr::Domain::Groom ? 1u :
        field.domain == expr::Domain::Primitive ? geometry.curveCount : geometry.pointCount;
    return field.count == expected && field.count <= size_t(std::numeric_limits<int>::max())
        ? StyleStatus::Ok : StyleStatus::InvalidArgument;
}

StyleStatus CudaNoise::validateBool(BoolField field, DeviceCurveGeometryView geometry,
                                    bool groomOnly, bool allowPrimitive) const {
    if (field.domain != expr::Domain::Groom && field.domain != expr::Domain::Primitive &&
        field.domain != expr::Domain::Point) return StyleStatus::InvalidArgument;
    if (groomOnly && field.domain != expr::Domain::Groom) return StyleStatus::InvalidArgument;
    if (!allowPrimitive && field.domain == expr::Domain::Primitive) return StyleStatus::InvalidArgument;
    if (field.domain == expr::Domain::Point) return StyleStatus::InvalidArgument;
    if (!field.data) return field.domain == expr::Domain::Groom && field.count == 0
        ? StyleStatus::Ok : ((field.domain == expr::Domain::Primitive && field.count == 0 &&
                              geometry.curveCount == 0) ? StyleStatus::Ok
                                                          : StyleStatus::InvalidArgument);
    const size_t expected = field.domain == expr::Domain::Groom ? 1u : geometry.curveCount;
    return field.count == expected && field.count <= size_t(std::numeric_limits<int>::max())
        ? StyleStatus::Ok : StyleStatus::InvalidArgument;
}

StyleStatus CudaNoise::validateInt(IntField field,
                                   DeviceCurveGeometryView geometry) const {
    if (field.domain != expr::Domain::Groom && field.domain != expr::Domain::Primitive &&
        field.domain != expr::Domain::Point) return StyleStatus::InvalidArgument;
    if (!field.data) {
        if (field.count != 0) return StyleStatus::InvalidArgument;
        if (field.domain != expr::Domain::Groom) {
            const size_t expected = field.domain == expr::Domain::Primitive
                ? geometry.curveCount : geometry.pointCount;
            return expected == 0 ? StyleStatus::Ok : StyleStatus::InvalidArgument;
        }
        return field.literal >= 1 && field.literal <= 6 ? StyleStatus::Ok : StyleStatus::InvalidValue;
    }
    const size_t expected = field.domain == expr::Domain::Groom ? 1u :
        field.domain == expr::Domain::Primitive ? geometry.curveCount : geometry.pointCount;
    return field.count == expected && field.count <= size_t(std::numeric_limits<int>::max())
        ? StyleStatus::Ok : StyleStatus::InvalidArgument;
}

static StyleStatus ValidateSeedField(IntField field, DeviceCurveGeometryView geometry) {
    if (field.domain != expr::Domain::Groom && field.domain != expr::Domain::Primitive)
        return StyleStatus::InvalidArgument;
    if (!field.data) {
        if (field.count != 0) return StyleStatus::InvalidArgument;
        if (field.domain == expr::Domain::Groom) return StyleStatus::Ok;
        return geometry.curveCount == 0 ? StyleStatus::Ok : StyleStatus::InvalidArgument;
    }
    const size_t expected = field.domain == expr::Domain::Groom ? 1u : geometry.curveCount;
    return field.count == expected && field.count <= size_t(std::numeric_limits<int>::max())
        ? StyleStatus::Ok : StyleStatus::InvalidArgument;
}

StyleStatus CudaNoise::begin(DeviceCurveGeometryView geometry,
                             DeviceView<float3> output, cudaStream_t stream,
                             UsdGenExecutionMemoryReservation* reservation) {
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return StyleStatus::CudaError;
    const cudaError_t errorAllocation = error_.reset(
        1, reservation, UsdGenExecutionResourceKind::Scratch);
    const cudaError_t stagingAllocation = errorAllocation == cudaSuccess
        ? staging_.reset(geometry.pointCount, reservation,
                         UsdGenExecutionResourceKind::Scratch)
        : cudaErrorMemoryAllocation;
    if (errorAllocation == cudaSuccess && stagingAllocation == cudaSuccess &&
        freshPreparing_ && !freshHostError_) {
        auto permit = TryReserveCudaExecutionBytes(
            sizeof(int), UsdGenExecutionResourceKind::Pinned, reservation);
        int* hostError = nullptr;
        if (!permit || cudaHostAlloc(reinterpret_cast<void**>(&hostError),
                         sizeof(int), cudaHostAllocDefault) != cudaSuccess)
            return StyleStatus::CudaError;
        freshHostError_ = hostError;
        freshHostErrorPermit_ = std::move(*permit);
    }
    const cudaError_t clearStatus = stagingAllocation == cudaSuccess
        ? cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) : cudaErrorMemoryAllocation;
    if (errorAllocation != cudaSuccess || stagingAllocation != cudaSuccess ||
        clearStatus != cudaSuccess) {
        // A failed asynchronous clear may follow work on this stream.  Fence
        // before releasing, and abandon allocations if completion is not
        // provable; Apply has not yet borrowed caller output at this point.
        if (CudaFenceStream(stream) != cudaSuccess) {
            staging_.quarantine();
            error_.quarantine();
            unprovenWork_ = true;
        } else {
            staging_.reset(0);
            error_.reset(0);
        }
        return StyleStatus::CudaError;
    }
    pointCount_ = geometry.pointCount;
    output_ = output;
    return StyleStatus::Ok;
}

StyleStatus CudaNoise::Apply(DeviceCurveGeometryView geometry,
                             DeviceView<const float> hairT, RestRootFrames frames,
                             NoiseParameters parameters, DeviceView<float3> output,
                             cudaStream_t stream,
                             UsdGenExecutionMemoryReservation* reservation) {
    if (pending_ || freshPending_ || unprovenWork_)
        return StyleStatus::InvalidArgument;
    int current = -1, streamDevice = -1;
    if (cudaGetDevice(&current) != cudaSuccess ||
        (stream && cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess) ||
        (stream && streamDevice != current) || (deviceIndex_ >= 0 && deviceIndex_ != current))
        return StyleStatus::InvalidArgument;
    if (deviceIndex_ < 0) deviceIndex_ = current;
    if (geometry.curveCount > size_t(std::numeric_limits<int>::max()) ||
        geometry.pointCount > size_t(std::numeric_limits<int>::max()) ||
        geometry.pointCount > size_t(std::numeric_limits<uint32_t>::max()) ||
        (geometry.curveCount == 0 && geometry.pointCount != 0) ||
        output.size != geometry.pointCount || (geometry.pointCount && !output.data) ||
        (!hairT.data && hairT.size != 0) ||
        (hairT.data && hairT.size != geometry.pointCount) ||
        (geometry.curveCount && (!geometry.points.data || !geometry.restPoints.data ||
            !geometry.curveOffsets.data || !geometry.stableIds.data ||
            geometry.points.size != geometry.pointCount ||
            geometry.restPoints.size != geometry.pointCount ||
            geometry.curveOffsets.size != geometry.curveCount + 1u ||
            geometry.stableIds.size != geometry.curveCount ||
            !frames.tangent.data || !frames.binormal.data || !frames.normal.data ||
            frames.tangent.size == 0 || frames.binormal.size != frames.tangent.size ||
            frames.normal.size != frames.tangent.size ||
            (!frames.stableIds.data && frames.tangent.size != geometry.curveCount) ||
            (frames.stableIds.data && frames.stableIds.size != frames.tangent.size))) ||
        (!geometry.curveCount && (geometry.points.size || geometry.restPoints.size ||
            geometry.curveOffsets.size > 1u ||
            (geometry.curveOffsets.size == 1u && !geometry.curveOffsets.data) ||
            geometry.stableIds.size || frames.tangent.size ||
            frames.binormal.size || frames.normal.size)) ||
        !parameters.magnitudeProfile.data || parameters.magnitudeProfile.size != kProfileSize)
        return StyleStatus::InvalidArgument;
    for (ScalarField field : {parameters.magnitude, parameters.frequency, parameters.correlation,
                              parameters.lacunarity, parameters.gain, parameters.preserveLength,
                              parameters.mask}) {
        const StyleStatus status = validateScalar(field, geometry);
        if (status != StyleStatus::Ok) return status;
    }
    StyleStatus status = StyleStatus::Ok;
    auto validateLiteralRange = [](ScalarField field, float minimum, float maximum,
                                   bool strictMinimum) {
        if (field.data || field.domain != expr::Domain::Groom) return StyleStatus::Ok;
        if (strictMinimum ? field.literal <= minimum : field.literal < minimum)
            return StyleStatus::InvalidValue;
        return field.literal > maximum ? StyleStatus::InvalidValue : StyleStatus::Ok;
    };
    const struct { ScalarField field; float minimum; float maximum; bool strict; } ranges[] = {
        {parameters.magnitude, 0.0f, std::numeric_limits<float>::max(), false},
        {parameters.frequency, 0.0f, std::numeric_limits<float>::max(), true},
        {parameters.correlation, 0.0f, 1.0f, false},
        {parameters.lacunarity, 1.0f, std::numeric_limits<float>::max(), true},
        {parameters.gain, 0.0f, 1.0f, false},
        {parameters.preserveLength, 0.0f, 1.0f, false},
        {parameters.mask, 0.0f, 1.0f, false}};
    for (auto const& entry : ranges) {
        status = validateLiteralRange(entry.field, entry.minimum, entry.maximum, entry.strict);
        if (status != StyleStatus::Ok) return status;
    }
    status = validateInt(parameters.octaves, geometry);
    if (status != StyleStatus::Ok) return status;
    status = ValidateSeedField(parameters.seed, geometry);
    if (status != StyleStatus::Ok) return status;
    status = validateBool(parameters.enabled, geometry, true, false);
    if (status != StyleStatus::Ok) return status;
    status = validateBool(parameters.cumulative, geometry, false, true);
    if (status != StyleStatus::Ok) return status;
    status = begin(geometry, output, stream, reservation);
    if (status != StyleStatus::Ok) return status;
    auto abortSubmission = [&] {
        if (CudaFenceStream(stream) != cudaSuccess) {
            staging_.quarantine();
            error_.quarantine();
            unprovenWork_ = true;
        }
        pointCount_ = 0;
        output_ = {};
        return StyleStatus::CudaError;
    };
    const size_t work = std::max(geometry.curveCount,
        std::max(geometry.pointCount, parameters.magnitudeProfile.size));
    ValidateInputs<<<Blocks(work), 256, 0, stream>>>(geometry, hairT, frames,
        parameters.magnitudeProfile, error_.data());
    if (cudaGetLastError() != cudaSuccess) return abortSubmission();
    if (geometry.curveCount) {
        NoiseKernel<<<Blocks(geometry.curveCount), 256, 0, stream>>>(
            geometry, hairT, frames, parameters, staging_.data(), error_.data());
        if (cudaGetLastError() != cudaSuccess) return abortSubmission();
    }
    if (cudaEventRecord(ready_, stream) != cudaSuccess) return abortSubmission();
    pending_ = true;
    return StyleStatus::Ok;
}

StyleStatus CudaNoise::finishPublication(cudaStream_t stream) {
    const cudaError_t copyStatus = pointCount_
        ? cudaMemcpyAsync(output_.data, staging_.data(),
                          pointCount_ * sizeof(float3), cudaMemcpyDeviceToDevice, stream)
        : cudaSuccess;
    // Fence unconditionally, including after a copy-submission error: work
    // before the failed call is still queued on this stream.
    const cudaError_t fenceStatus = CudaFenceStream(stream);
    if (copyStatus != cudaSuccess || fenceStatus != cudaSuccess) {
        // Do not leave a pending object holding borrowed caller output after
        // a failed publication fence.  An unprovable stream is quarantined.
        if (fenceStatus != cudaSuccess) {
            staging_.quarantine();
            error_.quarantine();
            unprovenWork_ = true;
        }
        pending_ = false;
        pointCount_ = 0;
        output_ = {};
        return StyleStatus::CudaError;
    }
    pending_ = false;
    pointCount_ = 0;
    output_ = {};
    return StyleStatus::Ok;
}

StyleStatus CudaNoise::Finish(cudaStream_t stream) {
    if (!pending_) return StyleStatus::InvalidArgument;
    int current = -1, streamDevice = -1, code = 0;
    if (cudaGetDevice(&current) != cudaSuccess || current != deviceIndex_ ||
        (stream && cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess) ||
        (stream && streamDevice != deviceIndex_)) return StyleStatus::InvalidArgument;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(&code, error_.data(), sizeof(code), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess) {
        if (CudaFenceStream(stream) != cudaSuccess) {
            staging_.quarantine();
            error_.quarantine();
            unprovenWork_ = true;
        }
        pending_ = false;
        pointCount_ = 0;
        output_ = {};
        return StyleStatus::CudaError;
    }
    if (code != 0) {
        pending_ = false;
        pointCount_ = 0;
        output_ = {};
        return DecodeError(code);
    }
    return finishPublication(stream);
}

StyleStatus CudaNoise::ApplyFresh(DeviceCurveGeometryView geometry,
                                  DeviceView<const float> hairT,
                                  RestRootFrames restFrames,
                                  NoiseParameters parameters,
                                  DeviceView<float3> output,
                                  cudaStream_t stream,
                                  UsdGenExecutionMemoryReservation* reservation) {
    if (pending_ || freshPending_ || freshPreparing_ || unprovenWork_)
        return StyleStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone)
        return StyleStatus::InvalidArgument;
    freshPreparing_ = true;
    StyleStatus const status = Apply(geometry, hairT, restFrames, parameters,
                                     output, stream, reservation);
    freshPreparing_ = false;
    if (status != StyleStatus::Ok) {
        if (unprovenWork_) freshUploadFailed_ = true;
        return status;
    }
    freshPending_ = true;
    freshCallbackArmed_ = false;
    freshUploadFailed_ = false;
    *freshHostError_ = std::numeric_limits<int>::min();
    return StyleStatus::Ok;
}

StyleStatus CudaNoise::FinishFreshAsync(
    cudaStream_t stream,
    void (*callback)(cudaStream_t, cudaError_t, void*) noexcept,
    void* userdata) {
    if (!pending_ || !freshPending_ || !callback || freshCallbackArmed_ ||
        freshUploadFailed_ || !freshHostError_)
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
        unprovenWork_ = true;
        return StyleStatus::CudaError;
    }
    if (pointCount_ && cudaMemcpyAsync(output_.data, staging_.data(),
            pointCount_ * sizeof(float3), cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        freshUploadFailed_ = true;
        unprovenWork_ = true;
        return StyleStatus::CudaError;
    }
    if (cudaMemcpyAsync(freshHostError_, error_.data(), sizeof(int),
                        cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        cudaStreamAddCallback(stream, callback, userdata, 0) != cudaSuccess) {
        freshUploadFailed_ = true;
        unprovenWork_ = true;
        return StyleStatus::CudaError;
    }
    freshCallbackArmed_ = true;
    return StyleStatus::Ok;
}

StyleStatus CudaNoise::CommitFreshFinish() {
    if (!pending_ || !freshPending_ || !freshCallbackArmed_ ||
        freshUploadFailed_ || !freshHostError_ ||
        *freshHostError_ == std::numeric_limits<int>::min())
        return StyleStatus::InvalidArgument;
    int const code = *freshHostError_;
    pending_ = false;
    freshPending_ = false;
    freshCallbackArmed_ = false;
    freshUploadFailed_ = false;
    unprovenWork_ = false;
    pointCount_ = 0;
    output_ = {};
    return code == 0 ? StyleStatus::Ok : DecodeError(code);
}

} // namespace usdGen::gpu
