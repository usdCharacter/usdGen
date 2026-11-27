#include "usdGenCurves/cuda/noise.h"
#include "api/testBackend.h"

#include "usdGenMath/usdGenMath/hash.h"

#include "SeExpr2/Noise.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(condition) do { \
    if (!(condition)) { \
        std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #condition, __FILE__, __LINE__); \
        return 1; \
    } \
} while (false)

namespace {

float3 V(float x, float y, float z) { return make_float3(x, y, z); }
float3 Add(float3 a, float3 b) { return V(a.x+b.x, a.y+b.y, a.z+b.z); }
float3 Sub(float3 a, float3 b) { return V(a.x-b.x, a.y-b.y, a.z-b.z); }
float3 Mul(float3 a, float f) { return V(a.x*f, a.y*f, a.z*f); }
float Length(float3 a) { return std::sqrt(a.x*a.x+a.y*a.y+a.z*a.z); }
bool Near(float a, float b, float tolerance = 2.0e-3f) {
    return std::fabs(a-b) <= tolerance;
}
bool Near(float3 a, float3 b, float tolerance = 2.0e-3f) {
    return Near(a.x,b.x,tolerance) && Near(a.y,b.y,tolerance) && Near(a.z,b.z,tolerance);
}
bool Finite(float3 a) {
    return std::isfinite(a.x) && std::isfinite(a.y) && std::isfinite(a.z);
}

void FreshDone(cudaStream_t, cudaError_t status, void* data) noexcept {
    static_cast<std::atomic<int>*>(data)->store(int(status), std::memory_order_release);
}

template <class T>
bool Upload(DeviceBuffer<T>& device, std::vector<T> const& host, cudaStream_t stream) {
    // Re-uploading equal-size negative fixtures must not invalidate the
    // DeviceView already borrowed by geometry/frames.
    return (device.size() == host.size() || device.reset(host.size()) == cudaSuccess) &&
        (!host.size() || cudaMemcpyAsync(device.data(), host.data(), host.size()*sizeof(T),
                                         cudaMemcpyHostToDevice, stream) == cudaSuccess);
}
template <class T>
bool Download(DeviceBuffer<T> const& device, std::vector<T>* host, cudaStream_t stream) {
    host->resize(device.size());
    return (!host->size() || cudaMemcpyAsync(host->data(), device.data(), host->size()*sizeof(T),
                                              cudaMemcpyDeviceToHost, stream) == cudaSuccess) &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}
template <class T>
DeviceView<const T> View(DeviceBuffer<T> const& buffer) {
    return {buffer.data(), buffer.size()};
}

float3 CpuVfbm(float3 restRoot, uint64_t id, int seed, float t,
               float frequency, float correlation, int octaves,
               float lacunarity, float gain) {
    const float3 hash = V(UsdGenDraw01(seed, id, kSaltNoise),
                          UsdGenDraw01(seed, id, kSaltNoise + 1u),
                          UsdGenDraw01(seed, id, kSaltNoise + 2u));
    // usdGen_seexpr exports the exact 3->3 double instantiation, but not a
    // float vector instantiation.  Call that named SeExpr primitive
    // directly (rather than relying on ExprBuiltins registration); CUDA is
    // compared with a tolerance for its deliberately float implementation.
    const double input[3] = {
        double(restRoot.x*correlation + hash.x*(1-correlation)),
        double(restRoot.y*correlation + hash.y*(1-correlation)),
        double(restRoot.z*correlation + hash.z*(1-correlation) + t*frequency)};
    double result[3]{};
    SeExpr2::FBM<3, 3, false, double>(input, result, octaves,
                                      double(lacunarity), double(gain));
    return V(float(result[0]), float(result[1]), float(result[2]));
}

float3 FrameVector(float3 tangent, float3 binormal, float3 normal, float3 local) {
    return Add(Add(Mul(tangent, local.x), Mul(binormal, local.y)), Mul(normal, local.z));
}

float Sample257(std::vector<float> const& values, float t) {
    const int lower = std::min(255, std::max(0, int(std::floor(t * 256.0f))));
    const float fraction = t * 256.0f - float(lower);
    return values[lower] + (values[lower + 1] - values[lower]) * fraction;
}

NoiseParameters Parameters(DeviceView<const float> magnitudeProfile) {
    NoiseParameters parameters;
    parameters.magnitudeProfile = magnitudeProfile;
    parameters.magnitude = ScalarField::Literal(.2f);
    parameters.frequency = ScalarField::Literal(1.7f);
    parameters.correlation = ScalarField::Literal(.4f);
    parameters.octaves = IntField::Literal(3);
    parameters.lacunarity = ScalarField::Literal(2.0f);
    parameters.gain = ScalarField::Literal(.5f);
    parameters.preserveLength = ScalarField::Literal(0.0f);
    parameters.seed = IntField::Literal(17);
    return parameters;
}

} // namespace

int main() {
    cudaStream_t producer = nullptr, consumer = nullptr;
    CHECK(cudaSetDevice(0) == cudaSuccess);
    CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) == cudaSuccess);

    {

    // Two ragged curves. The second frame is rotated, so matching the CPU
    // SeExpr vfbm oracle proves a vector field is projected through all three
    // root-frame axes rather than collapsed to a scalar root normal.
    const std::vector<float3> points{V(0,0,0), V(0,0,1), V(0,0,2),
                                     V(3,0,0), V(3,1,0), V(3,2,0), V(3,3,0)};
    const std::vector<float3> rest{V(0,0,0), V(0,0,1), V(0,0,2),
                                   V(3,0,0), V(3,1,0), V(3,2,0), V(3,3,0)};
    const std::vector<uint32_t> offsets{0u, 3u, 7u};
    const std::vector<uint64_t> ids{41u, 99u};
    const std::vector<float> hairT{0,.5f,1,0,.25f,.65f,1};
    const std::vector<float3> tangent{V(1,0,0), V(0,1,0)};
    const std::vector<float3> binormal{V(0,1,0), V(0,0,1)};
    const std::vector<float3> normal{V(0,0,1), V(1,0,0)};

    DeviceBuffer<float3> devicePoints, deviceRest, deviceTangent, deviceNormal, deviceBinormal, output;
    DeviceBuffer<uint32_t> deviceOffsets;
    DeviceBuffer<uint64_t> deviceIds;
    DeviceBuffer<float> deviceHairT, profile, pointMask, pointMagnitude;
    const std::vector<float> profileValues(257, 1.0f);
    CHECK(Upload(devicePoints, points, producer) && Upload(deviceRest, rest, producer) &&
          Upload(deviceOffsets, offsets, producer) && Upload(deviceIds, ids, producer) &&
          Upload(deviceHairT, hairT, producer) && Upload(deviceTangent, tangent, producer) &&
          Upload(deviceNormal, normal, producer) && Upload(deviceBinormal, binormal, producer) &&
          profile.reset(257) == cudaSuccess && output.reset(points.size()) == cudaSuccess);
    CHECK(cudaMemcpyAsync(profile.data(), profileValues.data(),
                          257*sizeof(float), cudaMemcpyHostToDevice, producer) == cudaSuccess);
    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);

    DeviceCurveGeometryView geometry{View(devicePoints), View(deviceRest), {}, View(deviceOffsets),
                                     View(deviceIds), 2, points.size()};
    RestRootFrames frames{
        View(deviceTangent), View(deviceBinormal), View(deviceNormal), {}};
    NoiseParameters parameters = Parameters(View(profile));
    CudaNoise noise;
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::Ok);
    std::vector<float3> raw;
    CHECK(Download(output, &raw, consumer));
    for (size_t c = 0; c < ids.size(); ++c) {
        for (uint32_t point = offsets[c]; point < offsets[c+1]; ++point) {
            const float3 local = Mul(CpuVfbm(rest[offsets[c]], ids[c], parameters.seed.literal, hairT[point],
                                              .0f + 1.7f, .4f, 3, 2.0f, .5f), .2f);
            CHECK(Near(raw[point], Add(points[point], FrameVector(tangent[c], binormal[c], normal[c], local))));
        }
    }

    // Fresh publication is copy-on-write: ApplyFresh may fill its private
    // staging but cannot touch the caller's candidate plane until the
    // explicit asynchronous finish protocol. The accepted input remains
    // byte-identical throughout, and all ordinary scratch/pinned accounting
    // returns to its baseline when the producer is destroyed.
    DeviceBuffer<float3> freshOutput;
    CHECK(freshOutput.reset(points.size()) == cudaSuccess);
    const std::vector<float3> freshSentinel(points.size(), V(91, 92, 93));
    CHECK(cudaMemcpyAsync(freshOutput.data(), freshSentinel.data(),
                          freshSentinel.size()*sizeof(float3),
                          cudaMemcpyHostToDevice, producer) == cudaSuccess &&
          cudaStreamSynchronize(producer) == cudaSuccess);
    auto resourcePool = FindUsdGenExecutionResourcePool(
        {usdGenTestUtils::Backend("cuda"), 0});
    CHECK(resourcePool);
    size_t const freshBaseline = resourcePool->Snapshot().usedBytes;
    {
        CudaNoise fresh;
        std::atomic<int> callbackStatus{int(cudaErrorNotReady)};
        parameters = Parameters(View(profile));
        CHECK(fresh.ApplyFresh(geometry, View(deviceHairT), frames, parameters,
                               freshOutput.view(), producer) == StyleStatus::Ok);
        std::vector<float3> beforePublish;
        CHECK(Download(freshOutput, &beforePublish, consumer));
        CHECK(std::memcmp(beforePublish.data(), freshSentinel.data(),
                          freshSentinel.size()*sizeof(float3)) == 0);
        CHECK(resourcePool->Snapshot().usedBytes > freshBaseline);
        CHECK(fresh.FinishFreshAsync(producer, FreshDone, &callbackStatus) ==
              StyleStatus::Ok);
        CHECK(cudaStreamSynchronize(producer) == cudaSuccess &&
              callbackStatus.load(std::memory_order_acquire) == int(cudaSuccess) &&
              fresh.CommitFreshFinish() == StyleStatus::Ok);
        CHECK(Download(freshOutput, &beforePublish, consumer));
        CHECK(std::memcmp(beforePublish.data(), freshSentinel.data(),
                          freshSentinel.size()*sizeof(float3)) != 0);
        std::vector<float3> acceptedInput;
        CHECK(Download(devicePoints, &acceptedInput, consumer));
        CHECK(std::memcmp(acceptedInput.data(), points.data(),
                          points.size()*sizeof(float3)) == 0);
    }
    CHECK(resourcePool->Snapshot().usedBytes == freshBaseline);
    {
        CudaNoise rejectedFresh;
        parameters = Parameters(View(profile));
        parameters.octaves = IntField::Literal(7);
        CHECK(rejectedFresh.ApplyFresh(geometry, View(deviceHairT), frames,
                                       parameters, freshOutput.view(), producer) ==
              StyleStatus::InvalidValue);
        // Validation occurred before any device or pinned candidate storage
        // was allocated, so a retry cannot consume the pool incrementally.
        CHECK(resourcePool->Snapshot().usedBytes == freshBaseline);
    }
    CHECK(resourcePool->Snapshot().usedBytes == freshBaseline);

    // Typed point float controls, point integer octaves, primitive seed, and
    // primitive cumulative booleans all remain device fields.
    DeviceBuffer<int32_t> pointOctaves, primitiveSeed;
    DeviceBuffer<unsigned char> cumulative;
    std::vector<float> magnitudeRamp(257, 1.0f);
    magnitudeRamp[0] = 0.0f;
    magnitudeRamp[128] = .5f;
    // usdGen:mask is the whole envelope and may be a point-domain field: the
    // former along-strand mask ramp is now just an expression over $t.
    const std::vector<float> pointMasks{0.0f,.75f,.5f,1.0f,.25f,.5f,1.0f};
    const std::vector<float> pointMagnitudes{.1f,.2f,.3f,.35f,.25f,.15f,.05f};
    CHECK(Upload(pointOctaves, std::vector<int32_t>{3,3,3,3,3,3,3}, producer) &&
          Upload(primitiveSeed, std::vector<int32_t>{17, 18}, producer) &&
          Upload(cumulative, std::vector<unsigned char>{1u,0u}, producer) &&
          Upload(pointMagnitude, pointMagnitudes, producer) &&
          Upload(pointMask, pointMasks, producer) && Upload(profile, magnitudeRamp, producer));
    parameters = Parameters(View(profile));
    parameters.magnitude = ScalarField::Device(View(pointMagnitude), expr::Domain::Point);
    parameters.octaves = IntField::Device(View(pointOctaves), expr::Domain::Point);
    parameters.seed = IntField::Device(View(primitiveSeed), expr::Domain::Primitive);
    parameters.mask = ScalarField::Device(View(pointMask), expr::Domain::Point);
    parameters.cumulative = BoolField::Device(
        {reinterpret_cast<uint8_t const*>(cumulative.data()), cumulative.size()}, expr::Domain::Primitive);
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::Ok && Download(output, &raw, consumer));
    // Exact §2.9 ordering: accumulate vfbm vectors before applying each
    // point's magnitude/ramp/mask.  The root has zero ramps, proving later
    // cumulative CVs still include its vector field.
    for (size_t c = 0; c < ids.size(); ++c) {
        float3 running = V(0,0,0);
        const bool isCumulative = c == 0;
        const int seed = c == 0 ? 17 : 18;
        for (uint32_t point = offsets[c]; point < offsets[c+1]; ++point) {
            const float3 field = CpuVfbm(rest[offsets[c]], ids[c], seed, hairT[point],
                                         1.7f, .4f, 3, 2, .5f);
            if (isCumulative) running = Add(running, field);
            const float scale = pointMagnitudes[point] * Sample257(magnitudeRamp, hairT[point]) *
                pointMasks[point];
            const float3 expected = Add(points[point], FrameVector(
                tangent[c], binormal[c], normal[c], Mul(isCumulative ? running : field, scale)));
            CHECK(Near(raw[point], expected));
        }
    }
    // Per-point seed has no coherent stable-id meaning and must fail closed.
    parameters.seed = IntField::Device(View(pointOctaves), expr::Domain::Point);
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::InvalidArgument);

    // Full restoration makes every nondegenerate output segment equal its
    // authored rest length. Capture the uncorrected result first so the
    // fractional case can independently reproduce the sequential rule.
    CHECK(Upload(profile, profileValues, producer));
    parameters = Parameters(View(profile));
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::Ok);
    std::vector<float3> unrestored;
    CHECK(Download(output, &unrestored, consumer));
    parameters.preserveLength = ScalarField::Literal(1.0f);
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::Ok);
    std::vector<float3> restored;
    CHECK(Download(output, &restored, consumer));
    for (size_t point = 1; point < restored.size(); ++point) {
        if (point == offsets[1]) continue;
        CHECK(Near(Length(Sub(restored[point], restored[point-1])),
                   Length(Sub(rest[point], rest[point-1])), 3.0e-3f));
    }
    parameters.preserveLength = ScalarField::Literal(.5f);
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::Ok);
    std::vector<float3> half;
    CHECK(Download(output, &half, consumer));
    for (size_t c = 0; c < ids.size(); ++c) {
        float3 previous = unrestored[offsets[c]];
        CHECK(Near(half[offsets[c]], previous));
        for (uint32_t point = offsets[c] + 1u; point < offsets[c + 1u]; ++point) {
            const float restLength = Length(Sub(rest[point], rest[point - 1u]));
            const float rawLength = Length(Sub(unrestored[point], previous));
            CHECK(restLength > 0.0f && rawLength > 0.0f);
            const float3 fullyRestored = Add(previous,
                Mul(Sub(unrestored[point], previous), restLength / rawLength));
            const float3 expected = Add(Mul(unrestored[point], .5f), Mul(fullyRestored, .5f));
            CHECK(Finite(half[point]) && Near(half[point], expected));
            previous = expected;
        }
    }

    // §0.5's zero resolved envelope is a bitwise input early-out even when
    // preserveLength would otherwise restore a visibly different rest shape.
    auto styledInput = points;
    styledInput[0] = V(-0.0f, 0.0f, 0.0f); // preserve the signed-zero byte.
    styledInput[1] = V(4.0f, 0.0f, 0.0f);
    styledInput[2] = V(7.0f, 0.0f, 0.0f);
    CHECK(Upload(devicePoints, styledInput, producer));
    parameters = Parameters(View(profile));
    parameters.mask = ScalarField::Literal(0.0f);
    parameters.preserveLength = ScalarField::Literal(1.0f);
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::Ok && Download(output, &raw, consumer));
    CHECK(std::memcmp(raw.data(), styledInput.data(), styledInput.size()*sizeof(float3)) == 0);
    CHECK(Upload(devicePoints, points, producer));

    // Disabled is an exact pass-through, and a bad frame, non-finite input,
    // or malformed offsets leave
    // the caller's output untouched because only staging is ever published.
    parameters = Parameters(View(profile));
    parameters.enabled = BoolField::Literal(false);
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::Ok && Download(output, &raw, consumer));
    CHECK(std::memcmp(raw.data(), points.data(), points.size()*sizeof(float3)) == 0);
    const std::vector<float3> sentinel(points.size(), V(71,71,71));
    CHECK(cudaMemcpyAsync(output.data(), sentinel.data(), sentinel.size()*sizeof(float3),
                          cudaMemcpyHostToDevice, producer) == cudaSuccess);
    CHECK(Upload(deviceBinormal, std::vector<float3>{V(0,1,0), V(0,1,0)}, producer));
    parameters = Parameters(View(profile));
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::InvalidValue && Download(output, &raw, consumer));
    CHECK(std::memcmp(raw.data(), sentinel.data(), sentinel.size()*sizeof(float3)) == 0);
    CHECK(Upload(deviceBinormal, binormal, producer));
    // Noise3 floors float coordinates to an int lattice. Finite-but-huge
    // rest roots and a later fBm octave must reject before that conversion;
    // neither may publish staging into the caller's sentinel output.
    auto hugeRest = rest;
    hugeRest[0].x = std::numeric_limits<float>::max();
    CHECK(Upload(deviceRest, hugeRest, producer));
    parameters = Parameters(View(profile));
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::InvalidValue && Download(output, &raw, consumer));
    CHECK(std::memcmp(raw.data(), sentinel.data(), sentinel.size()*sizeof(float3)) == 0);
    CHECK(Upload(deviceRest, rest, producer));
    parameters = Parameters(View(profile));
    parameters.octaves = IntField::Literal(2);
    parameters.lacunarity = ScalarField::Literal(std::numeric_limits<float>::max());
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::InvalidValue && Download(output, &raw, consumer));
    CHECK(std::memcmp(raw.data(), sentinel.data(), sentinel.size()*sizeof(float3)) == 0);
    std::vector<float3> nonFinitePoints = points;
    nonFinitePoints[4].x = std::numeric_limits<float>::quiet_NaN();
    CHECK(Upload(devicePoints, nonFinitePoints, producer));
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::NonFiniteInput && Download(output, &raw, consumer));
    CHECK(std::memcmp(raw.data(), sentinel.data(), sentinel.size()*sizeof(float3)) == 0);
    CHECK(Upload(devicePoints, points, producer));
    // A terminal empty curve used to pass terminal-offset-only validation.
    CHECK(Upload(deviceOffsets, std::vector<uint32_t>{0u,3u,3u}, producer));
    CHECK(noise.Apply(geometry, View(deviceHairT), frames, parameters,
                      {output.data(), output.size()}, producer) == StyleStatus::Ok);
    CHECK(noise.Finish(consumer) == StyleStatus::InvalidArgument && Download(output, &raw, consumer));
    CHECK(std::memcmp(raw.data(), sentinel.data(), sentinel.size()*sizeof(float3)) == 0);
    CHECK(Upload(deviceOffsets, offsets, producer));

    DeviceCurveGeometryView empty{{}, {}, {}, {}, {}, 0, 0};
    RestRootFrames emptyFrames{};
    CudaNoise emptyNoise;
    NoiseParameters emptyParameters = Parameters(View(profile));
    // This is the real evaluator representation for empty typed fields: no
    // invented groom literal and no data allocation for zero-sized domains.
    emptyParameters.magnitude = ScalarField::Device(DeviceView<const float>{}, expr::Domain::Point);
    emptyParameters.frequency = ScalarField::Device(DeviceView<const float>{}, expr::Domain::Primitive);
    emptyParameters.correlation = ScalarField::Device(DeviceView<const float>{}, expr::Domain::Point);
    emptyParameters.lacunarity = ScalarField::Device(DeviceView<const float>{}, expr::Domain::Primitive);
    emptyParameters.gain = ScalarField::Device(DeviceView<const float>{}, expr::Domain::Point);
    emptyParameters.preserveLength = ScalarField::Device(DeviceView<const float>{}, expr::Domain::Primitive);
    emptyParameters.mask = ScalarField::Device(DeviceView<const float>{}, expr::Domain::Primitive);
    emptyParameters.octaves = IntField::Device(DeviceView<const int32_t>{}, expr::Domain::Point);
    emptyParameters.seed = IntField::Device(DeviceView<const int32_t>{}, expr::Domain::Primitive);
    emptyParameters.cumulative = BoolField::Device(DeviceView<const uint8_t>{}, expr::Domain::Primitive);
    CHECK(emptyNoise.Apply(empty, {}, emptyFrames, emptyParameters, {}, producer) == StyleStatus::Ok);
    CHECK(emptyNoise.Finish(consumer) == StyleStatus::Ok);

    // C3 Source/Grow publishes the canonical empty topology with one zero
    // terminal offset. Exercise the fresh asynchronous publication protocol,
    // not merely the legacy zero-offset-array representation above.
    DeviceBuffer<uint32_t> emptyOffsets;
    CHECK(Upload(emptyOffsets, std::vector<uint32_t>{0u}, producer));
    empty.curveOffsets = View(emptyOffsets);
    {
        CudaNoise freshEmpty;
        std::atomic<int> completion{int(cudaErrorNotReady)};
        CHECK(freshEmpty.ApplyFresh(empty, {}, emptyFrames, emptyParameters, {}, producer) == StyleStatus::Ok);
        CHECK(freshEmpty.FinishFreshAsync(producer, FreshDone, &completion) == StyleStatus::Ok);
        CHECK(cudaStreamSynchronize(producer) == cudaSuccess &&
              completion.load(std::memory_order_acquire) == int(cudaSuccess));
        CHECK(freshEmpty.CommitFreshFinish() == StyleStatus::Ok);
    }
    // Even with no curves to launch, a nonzero device-side terminal offset
    // must fail validation. Invalid host shapes must fail before submission.
    CHECK(Upload(emptyOffsets, std::vector<uint32_t>{1u}, producer));
    CHECK(emptyNoise.Apply(empty, {}, emptyFrames, emptyParameters, {}, producer) == StyleStatus::Ok);
    CHECK(emptyNoise.Finish(consumer) == StyleStatus::InvalidArgument);
    empty.curveOffsets = {nullptr, 1};
    CHECK(emptyNoise.ApplyFresh(empty, {}, emptyFrames, emptyParameters, {}, producer) == StyleStatus::InvalidArgument);
    CHECK(Upload(emptyOffsets, std::vector<uint32_t>{0u, 0u}, producer));
    empty.curveOffsets = View(emptyOffsets);
    CHECK(emptyNoise.ApplyFresh(empty, {}, emptyFrames, emptyParameters, {}, producer) == StyleStatus::InvalidArgument);
    CHECK(cudaStreamSynchronize(producer) == cudaSuccess);

    } // All CUDA buffers/events are destroyed before their streams.

    CHECK(cudaStreamDestroy(producer) == cudaSuccess);
    CHECK(cudaStreamDestroy(consumer) == cudaSuccess);
    return 0;
}
