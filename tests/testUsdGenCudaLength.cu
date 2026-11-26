#include "usdGenCurves/cuda/length.h"
#include "../libs/base/usdGenMath/usdGenMath/hash.h"
#include "floatUlpFixture.h"

#include <cmath>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <vector>
#include <limits>
#include <algorithm>

using namespace usdGen;
using namespace usdGen::gpu;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "CHECK failed: %s (%s:%d)\n", #x, __FILE__, __LINE__); return 1; } } while (false)

template <class T>
static bool Upload(DeviceBuffer<T>& d, std::vector<T> const& values, cudaStream_t s) {
    return cudaMemcpyAsync(d.data(), values.data(), values.size()*sizeof(T),
                           cudaMemcpyHostToDevice, s) == cudaSuccess;
}
template <class T>
static bool Download(DeviceBuffer<T> const& d, std::vector<T>& values, cudaStream_t s) {
    if (!values.empty() && cudaMemcpyAsync(values.data(), d.data(),
                                           values.size()*sizeof(T),
                                           cudaMemcpyDeviceToHost, s) != cudaSuccess)
        return false;
    return cudaStreamSynchronize(s) == cudaSuccess;
}
static bool Near(float a, float b) { return std::fabs(a-b) < 3.0e-4f; }
static DeviceView<const float> Const(DeviceBuffer<float> const& d) {
    return {d.data(), d.size()};
}
static DeviceCurveGeometryView Geometry(DeviceBuffer<float3> const& points,
                                        DeviceBuffer<uint32_t> const& offsets,
                                        DeviceBuffer<uint64_t> const& ids) {
    return {{points.data(), points.size()}, {}, {},
            {offsets.data(), offsets.size()}, {ids.data(), ids.size()}, 2, 5};
}

struct FreshSignal {
    std::atomic<int> calls{0};
    std::atomic<int> status{int(cudaErrorUnknown)};
};
static void FreshCallback(cudaStream_t, cudaError_t status, void* userdata) noexcept {
    auto* signal=static_cast<FreshSignal*>(userdata);
    signal->status.store(int(status),std::memory_order_release);
    signal->calls.fetch_add(1,std::memory_order_release);
}

int main() {
    cudaStream_t producer = nullptr, consumer = nullptr;
    CHECK(cudaStreamCreateWithFlags(&producer, cudaStreamNonBlocking) == cudaSuccess);
    CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) == cudaSuccess);

    DeviceBuffer<float3> points, output;
    DeviceBuffer<uint32_t> offsets;
    DeviceBuffer<uint64_t> ids;
    DeviceBuffer<uint8_t> keep;
    CHECK(points.reset(5) == cudaSuccess && output.reset(5) == cudaSuccess &&
          offsets.reset(3) == cudaSuccess && ids.reset(2) == cudaSuccess &&
          keep.reset(2) == cudaSuccess);
    const std::vector<float3> source{{0,0,0}, {1,0,0}, {1,1,0},
                                     {0,0,0}, {0,0,3}};
    CHECK(Upload(points, source, producer) &&
          Upload(offsets, {0u,3u,5u}, producer) &&
          Upload(ids, {17u,29u}, producer) && cudaStreamSynchronize(producer) == cudaSuccess);
    const auto geometry = Geometry(points, offsets, ids);
    std::vector<float3> got(5);
    std::vector<uint8_t> gotKeep(2);
    CudaLength op;

    // Scale is root anchored, supports ragged 3/2-CV curves, and does not
    // publish into the borrowed output until Finish on another stream.
    LengthParameters p;
    p.value = ScalarField::Literal(2.0f);
    CHECK(cudaMemsetAsync(output.data(), 0xA5, output.size()*sizeof(float3), producer) == cudaSuccess);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok);
    CHECK(op.pending() && op.Finish(consumer) == StyleStatus::Ok);
    CHECK(Download(output, got, consumer) && Download(keep, gotKeep, consumer));
    CHECK(Near(got[0].x,0) && Near(got[1].x,2) && Near(got[2].x,2) && Near(got[2].y,2));
    CHECK(Near(got[3].z,0) && Near(got[4].z,6) && gotKeep[0] == 1 && gotKeep[1] == 1);

    // Exact random draw and swapped random endpoints are deterministic.
    p.value = ScalarField::Literal(1.0f);
    p.random = Vec2Field::Literal(make_float2(2.0f, 1.0f));
    p.seed = 11;
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok);
    CHECK(op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer));
    // The endpoints are swapped in the same way as the authored CPU schema.
    CHECK(Near(got[2].y, 1 + UsdGenDraw01(11, 17, usdGen::kSaltLength)));
    CHECK(Near(got[4].z, 3 * (1 + UsdGenDraw01(11, 29, usdGen::kSaltLength))));

    // Unit lines expose the pinned draw directly as the tip coordinate: require
    // bit-exact equality, not a geometry tolerance that could hide hash drift.
    {
        constexpr uint32_t count = 128;
        std::vector<float3> unitPoints(2 * count);
        std::vector<uint32_t> unitOffsets(count + 1);
        std::vector<uint64_t> unitIds(count);
        for (uint32_t i = 0; i < count; ++i) {
            unitPoints[2*i] = make_float3(0,0,0);
            unitPoints[2*i+1] = make_float3(1,0,0);
            unitOffsets[i] = 2*i;
            unitIds[i] = (uint64_t(i * 2654435761u) << 32) | (0xffffffffu-i);
        }
        unitOffsets[count] = 2*count;
        unitIds[0] = 0;
        unitIds[1] = UINT64_MAX;
        unitIds[2] = uint64_t(1) << 63;
        DeviceBuffer<float3> unitInput, unitOutput;
        DeviceBuffer<uint32_t> unitOffsetBuffer;
        DeviceBuffer<uint64_t> unitIdBuffer;
        DeviceBuffer<uint8_t> unitKeep;
        CHECK(unitInput.reset(2*count) == cudaSuccess &&
              unitOutput.reset(2*count) == cudaSuccess &&
              unitOffsetBuffer.reset(count+1) == cudaSuccess &&
              unitIdBuffer.reset(count) == cudaSuccess &&
              unitKeep.reset(count) == cudaSuccess);
        CHECK(Upload(unitInput, unitPoints, producer) &&
              Upload(unitOffsetBuffer, unitOffsets, producer));
        DeviceCurveGeometryView unitGeometry{
            {unitInput.data(), unitInput.size()}, {}, {},
            {unitOffsetBuffer.data(), unitOffsetBuffer.size()},
            {unitIdBuffer.data(), unitIdBuffer.size()}, count, 2*count};
        std::vector<float3> unitGot(2*count);
        std::vector<uint8_t> unitGotKeep(count);
        for (int permutation = 0; permutation < 2; ++permutation) {
            if (permutation) std::reverse(unitIds.begin(), unitIds.end());
            CHECK(Upload(unitIdBuffer, unitIds, producer));
            for (int seed : {0, -1, 11, std::numeric_limits<int>::min(),
                             std::numeric_limits<int>::max()}) {
                LengthParameters randomParameters;
                randomParameters.value = ScalarField::Literal(1.f);
                randomParameters.random = Vec2Field::Literal(make_float2(0.f,1.f));
                randomParameters.seed = seed;
                CHECK(op.Apply(unitGeometry, {}, randomParameters, unitOutput.view(),
                               unitKeep.view(), producer) == StyleStatus::Ok);
                CHECK(op.Finish(consumer) == StyleStatus::Ok &&
                      Download(unitOutput, unitGot, consumer) &&
                      Download(unitKeep, unitGotKeep, consumer));
                for (uint32_t i = 0; i < count; ++i) {
                    float const draw = UsdGenDraw01(seed, unitIds[i], usdGen::kSaltLength);
                    CHECK(std::memcmp(&unitGot[2*i+1].x, &draw, sizeof(float)) == 0);
                    CHECK(unitGot[2*i].x == 0 && unitGot[2*i].y == 0 &&
                          unitGot[2*i].z == 0 && unitGot[2*i+1].y == 0 &&
                          unitGot[2*i+1].z == 0 && unitGotKeep[i] == 1);
                }
            }
        }
        CHECK(Download(unitInput, unitGot, consumer));
        CHECK(std::memcmp(unitGot.data(), unitPoints.data(), unitPoints.size()*sizeof(float3)) == 0);
    }

    // A reversed unit line exposes tiny positive envelopes as subnormal output
    // coordinates. Zero-underflow must instead be a byte-exact topology no-op.
    {
        DeviceBuffer<float3> tinyInput, tinyOutput;
        DeviceBuffer<uint32_t> tinyOffsets;
        DeviceBuffer<uint64_t> tinyIds;
        DeviceBuffer<uint8_t> tinyKeep;
        CHECK(tinyInput.reset(2) == cudaSuccess && tinyOutput.reset(2) == cudaSuccess &&
              tinyOffsets.reset(2) == cudaSuccess && tinyIds.reset(1) == cudaSuccess &&
              tinyKeep.reset(1) == cudaSuccess);
        std::vector<float3> tinySource{{1,0,0},{0,0,0}}, tinyGot(2);
        std::vector<uint8_t> tinyGotKeep(1);
        CHECK(Upload(tinyInput, tinySource, producer) &&
              Upload(tinyOffsets, {0u,2u}, producer) && Upload(tinyIds, {uint64_t(7)}, producer));
        DeviceCurveGeometryView tinyGeometry{{tinyInput.data(),2}, {}, {},
            {tinyOffsets.data(),2}, {tinyIds.data(),1}, 1, 2};
        LengthParameters tiny;
        tiny.value = ScalarField::Literal(2.f);
        // A subnormal envelope must survive as a subnormal, not flush to zero.
        tiny.mask = ScalarField::Literal(std::numeric_limits<float>::min() * .5f);
        CHECK(op.Apply(tinyGeometry, {}, tiny, tinyOutput.view(), tinyKeep.view(), producer) == StyleStatus::Ok &&
              op.Finish(consumer) == StyleStatus::Ok && Download(tinyOutput, tinyGot, consumer));
        float const subnormal = -std::numeric_limits<float>::min()*.5f;
        CHECK(std::memcmp(&tinyGot[1].x, &subnormal, sizeof(float)) == 0);
        // An exact zero envelope is a bitwise pass-through of both geometry
        // and topology, whatever the length bounds say.
        tiny.mask = ScalarField::Literal(0.f);
        tiny.minRemainingLength = ScalarField::Literal(100.f);
        tiny.cullThreshold = ScalarField::Literal(1000.f);
        CHECK(op.Apply(tinyGeometry, {}, tiny, tinyOutput.view(), tinyKeep.view(), producer) == StyleStatus::Ok &&
              op.Finish(consumer) == StyleStatus::Ok && Download(tinyOutput, tinyGot, consumer) &&
              Download(tinyKeep, tinyGotKeep, consumer));
        CHECK(std::memcmp(tinyGot.data(), tinySource.data(), sizeof(float3)*2) == 0 &&
              tinyGotKeep == std::vector<uint8_t>({1}));
        tiny.mode = LengthMode::Cull;
        CHECK(op.Apply(tinyGeometry, {}, tiny, tinyOutput.view(), tinyKeep.view(), producer) == StyleStatus::Ok &&
              op.Finish(consumer) == StyleStatus::Ok && Download(tinyOutput, tinyGot, consumer) &&
              Download(tinyKeep, tinyGotKeep, consumer));
        CHECK(std::memcmp(tinyGot.data(), tinySource.data(), sizeof(float3)*2) == 0 &&
              tinyGotKeep == std::vector<uint8_t>({1}));
    }

    // Set + cutExtend keepParam cuts and collapses trailing CVs; reparam
    // distributes CVs and extends along the last non-degenerate tangent.
    p = LengthParameters{};
    p.mode = LengthMode::Set; p.method = LengthMethod::CutExtend;
    p.rebuild = LengthRebuild::KeepParam; p.value = ScalarField::Literal(.5f);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer));
    CHECK(Near(got[1].x,.5f) && Near(got[2].x,.5f) && Near(got[2].y,0));
    p.rebuild = LengthRebuild::Reparam; p.value = ScalarField::Literal(4.0f);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer));
    CHECK(Near(got[1].x,1) && Near(got[1].y,1) && Near(got[2].y,3) && Near(got[4].z,4));

    // Point fields are accepted and evaluated at every CV; primitive fields
    // broadcast per curve. A zero envelope is an exact no-op and keep=1.
    DeviceBuffer<float> pointBlend, primitiveThreshold;
    CHECK(pointBlend.reset(5) == cudaSuccess && primitiveThreshold.reset(2) == cudaSuccess);
    CHECK(Upload(pointBlend, {0,0,1,1,1}, producer) &&
          Upload(primitiveThreshold, {1.5f,3.1f}, producer) &&
          cudaStreamSynchronize(producer) == cudaSuccess);
    p = LengthParameters{};
    p.value = ScalarField::Literal(2.0f);
    p.mask = ScalarField::Device(Const(pointBlend), expr::Domain::Point);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer) &&
          Download(keep, gotKeep, consumer));
    CHECK(Near(got[0].x,0) && Near(got[1].x,1) && Near(got[2].x,2));
    p = LengthParameters{}; p.mode = LengthMode::Cull;
    p.cullThreshold = ScalarField::Device(Const(primitiveThreshold), expr::Domain::Primitive);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer) &&
          Download(keep, gotKeep, consumer));
    CHECK(std::memcmp(got.data(), source.data(), sizeof(source[0])*source.size()) == 0 &&
          gotKeep[0] == 1 && gotKeep[1] == 0);

    // Disabled and all-zero envelopes preserve both geometry and topology.
    p = LengthParameters{}; p.value = ScalarField::Literal(9);
    p.enabled = BoolField::Literal(false);
    p.cullThreshold = ScalarField::Literal(100);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer) &&
          Download(keep, gotKeep, consumer));
    CHECK(std::memcmp(got.data(), source.data(), sizeof(source[0])*source.size()) == 0 &&
          gotKeep[0] == 1 && gotKeep[1] == 1);

    DeviceBuffer<float> pointValue;
    CHECK(pointValue.reset(5) == cudaSuccess);
    CHECK(Upload(pointValue, {0,1,2,0,1}, producer));
    p = LengthParameters{}; p.mode = LengthMode::Set;
    p.value = ScalarField::Device(Const(pointValue), expr::Domain::Point);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer));
    CHECK(Near(got[1].x,.5f) && Near(got[2].y,1) && Near(got[4].z,1));
    // Scale + cutExtend must use each CV's expression result, not the root's.
    CHECK(Upload(pointValue, {0,.5f,2,0,1}, producer));
    p.mode = LengthMode::Scale; p.method = LengthMethod::CutExtend;
    p.rebuild = LengthRebuild::Reparam;
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer));
    CHECK(Near(got[1].x,.5f) && Near(got[1].y,0) && Near(got[2].y,3));
    p = LengthParameters{}; p.value = ScalarField::Literal(.1f);
    p.minRemainingLength = ScalarField::Literal(2.5f);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer));
    CHECK(Near(got[2].x,1.25f) && Near(got[2].y,1.25f) && Near(got[4].z,2.5f));
    // Binding floor on both ragged curves: their current arcs are 2 and 3.
    // These hand-derived complete arrays distinguish radial, keepParam, and
    // Reparam, rather than merely checking whether the dispatch succeeded.
    const std::vector<float3> minimumRadial{{0,0,0},{1.25f,0,0},{1.25f,1.25f,0},{0,0,0},{0,0,2.5f}};
    const std::vector<float3> minimumKeep{{0,0,0},{1,0,0},{1,1.5f,0},{0,0,0},{0,0,2.5f}};
    const std::vector<float3> minimumReparam{{0,0,0},{1,.25f,0},{1,1.5f,0},{0,0,0},{0,0,2.5f}};
    // Partial scalar envelopes blend the edited curve back onto the original,
    // not the target length; bent Reparam makes that distinction observable.
    for (unsigned method = 0; method < 3; ++method) {
        LengthParameters envelope;
        envelope.mode = LengthMode::Set;
        envelope.value = ScalarField::Literal(4.f);
        envelope.mask = ScalarField::Literal(.25f);
        envelope.method = method ? LengthMethod::CutExtend : LengthMethod::Scale;
        envelope.rebuild = method == 2 ? LengthRebuild::Reparam : LengthRebuild::KeepParam;
        CHECK(op.Apply(geometry, {}, envelope, output.view(), keep.view(), producer) == StyleStatus::Ok &&
              op.Finish(consumer) == StyleStatus::Ok &&
              Download(output, got, consumer) && Download(keep, gotKeep, consumer));
        std::vector<float3> expected = method == 0
            ? std::vector<float3>{{0,0,0},{1.25f,0,0},{1.25f,1.25f,0},{0,0,0},{0,0,3.25f}}
            : method == 1
            ? std::vector<float3>{{0,0,0},{1,0,0},{1,1.5f,0},{0,0,0},{0,0,3.25f}}
            : std::vector<float3>{{0,0,0},{1,.25f,0},{1,1.5f,0},{0,0,0},{0,0,3.25f}};
        CHECK(FloatBytesWithinUlps(got.data(), expected.data(), expected.size()*sizeof(float3), 4) &&
              gotKeep == std::vector<uint8_t>({1,1}));
    }
    for (auto mode : {LengthMode::Scale, LengthMode::Set}) {
        for (auto method : {LengthMethod::Scale, LengthMethod::CutExtend}) {
            for (auto rebuild : {LengthRebuild::KeepParam, LengthRebuild::Reparam}) {
                p = LengthParameters{}; p.mode = mode; p.method = method; p.rebuild = rebuild;
                p.value = ScalarField::Literal(mode == LengthMode::Scale ? .25f : 1.f);
                p.minRemainingLength = ScalarField::Literal(2.5f);
                CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
                      op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer) &&
                      Download(keep, gotKeep, consumer));
                auto const& expected = method == LengthMethod::Scale ? minimumRadial :
                    rebuild == LengthRebuild::KeepParam ? minimumKeep : minimumReparam;
                CHECK(FloatBytesWithinUlps(got.data(), expected.data(), expected.size()*sizeof(float3), 4));
                CHECK(gotKeep == std::vector<uint8_t>({1,1}));
            }
        }
    }
    // Actual Reparam output arc is sqrt(1.0625)+1.25, less than the 2.5 target.
    p.cullThreshold = ScalarField::Literal(2.4f);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer) &&
          Download(keep, gotKeep, consumer));
    CHECK(FloatBytesWithinUlps(got.data(), minimumReparam.data(), got.size()*sizeof(float3), 4) &&
          gotKeep == std::vector<uint8_t>({0,1}));
    // Finite inputs may overflow the staged Scale product before a zero
    // multiplier. CUDA fmaxf recovers the finite floor from that raw NaN.
    p = LengthParameters{};
    p.value = ScalarField::Literal(std::numeric_limits<float>::max());
    p.random = Vec2Field::Literal(make_float2(0.f, 0.f));
    p.minRemainingLength = ScalarField::Literal(2.5f);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok &&
          Download(output, got, consumer) && Download(keep, gotKeep, consumer));
    CHECK(FloatBytesWithinUlps(got.data(), minimumRadial.data(), got.size()*sizeof(float3), 4) &&
          gotKeep == std::vector<uint8_t>({1,1}));

    // A nonbinding floor does not cap an independently larger requested target.
    p = LengthParameters{}; p.mode = LengthMode::Set;
    p.value = ScalarField::Literal(4); p.minRemainingLength = ScalarField::Literal(1);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer));
    const std::vector<float3> nonbindingMinimum{{0,0,0},{2,0,0},{2,2,0},{0,0,0},{0,0,4}};
    CHECK(FloatBytesWithinUlps(got.data(), nonbindingMinimum.data(), got.size()*sizeof(float3), 4));
    p = LengthParameters{}; p.mode = LengthMode::Cull;
    p.minRemainingLength = ScalarField::Literal(1000.0f);
    p.cullThreshold = ScalarField::Literal(2.5f);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(output, got, consumer) &&
          Download(keep, gotKeep, consumer));
    CHECK(std::memcmp(got.data(), source.data(), source.size()*sizeof(float3)) == 0 &&
          gotKeep == std::vector<uint8_t>({0,1}));
    const std::vector<unsigned char> minimumSentinel(got.size()*sizeof(float3), 0xA5);
    for (float minimum : {-1.f, NAN, INFINITY}) {
        CHECK(cudaMemsetAsync(output.data(), 0xA5, output.size()*sizeof(float3), producer) == cudaSuccess &&
              cudaMemsetAsync(keep.data(), 0xA5, keep.size(), producer) == cudaSuccess &&
              cudaStreamSynchronize(producer) == cudaSuccess);
        p = LengthParameters{}; p.minRemainingLength = ScalarField::Literal(minimum);
        auto status = op.Apply(geometry, {}, p, output.view(), keep.view(), producer);
        if (std::isfinite(minimum)) CHECK(status == StyleStatus::Ok && op.Finish(consumer) == StyleStatus::InvalidValue);
        else CHECK(status == StyleStatus::NonFiniteInput && !op.pending());
        CHECK(Download(output, got, consumer) && Download(keep, gotKeep, consumer) &&
              std::memcmp(got.data(), minimumSentinel.data(), minimumSentinel.size()) == 0 &&
              gotKeep == std::vector<uint8_t>({0xA5,0xA5}));
    }
    DeviceBuffer<float3> zeroPoints;
    CHECK(zeroPoints.reset(5) == cudaSuccess &&
          Upload(zeroPoints, std::vector<float3>(5, make_float3(0,0,0)), producer) &&
          cudaStreamSynchronize(producer) == cudaSuccess);
    auto zeroGeometry = Geometry(zeroPoints, offsets, ids);
    CHECK(cudaMemsetAsync(output.data(), 0xA5, output.size()*sizeof(float3), producer) == cudaSuccess &&
          cudaMemsetAsync(keep.data(), 0xA5, keep.size()*sizeof(uint8_t), producer) == cudaSuccess);
    p = LengthParameters{}; p.minRemainingLength = ScalarField::Literal(1.0f);
    p.value = ScalarField::Literal(0);
    CHECK(op.Apply(zeroGeometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::InvalidValue && Download(output, got, consumer) &&
          Download(keep, gotKeep, consumer));
    CHECK(std::memcmp(got.data(), minimumSentinel.data(), minimumSentinel.size()) == 0 &&
          gotKeep == std::vector<uint8_t>({0xA5,0xA5}));
    std::vector<float3> minimumSourceAgain(5);
    CHECK(Download(points, minimumSourceAgain, consumer) &&
          std::memcmp(minimumSourceAgain.data(), source.data(), source.size()*sizeof(float3)) == 0);
    p = LengthParameters{}; p.mode = LengthMode::Cull;
    p.cullThreshold = ScalarField::Literal(100);
    p.mask = ScalarField::Literal(0.f);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::Ok && Download(keep, gotKeep, consumer));
    CHECK(gotKeep == std::vector<uint8_t>({1,1}));

    // Device validation catches malformed offsets and nonfinite input without
    // publishing staging. Host field-count validation queues no work.
    CHECK(Upload(offsets, {0u,3u,9u}, producer) &&
          cudaMemsetAsync(output.data(), 0x5A, output.size()*sizeof(float3), producer) == cudaSuccess);
    CHECK(op.Apply(geometry, {}, LengthParameters{}, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::InvalidArgument);
    CHECK(Download(output, got, consumer));
    std::vector<unsigned char> sentinel(sizeof(float3)*got.size(), 0x5A);
    CHECK(std::memcmp(got.data(), sentinel.data(), sentinel.size()) == 0);
    CHECK(Upload(offsets, {0u,3u,5u}, producer));
    std::vector<float3> bad = source; bad[3].x = NAN;
    CHECK(Upload(points, bad, producer));
    CHECK(op.Apply(geometry, {}, LengthParameters{}, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          op.Finish(consumer) == StyleStatus::NonFiniteInput);
    CHECK(Download(output, got, consumer));
    CHECK(std::memcmp(got.data(), sentinel.data(), sentinel.size()) == 0);
    CHECK(LengthParameters{}.value.domain == expr::Domain::Groom);
    p = LengthParameters{};
    DeviceBuffer<float> shortField; CHECK(shortField.reset(1) == cudaSuccess);
    p.value = ScalarField::Device(Const(shortField), expr::Domain::Primitive);
    CHECK(op.Apply(geometry, {}, p, output.view(), keep.view(), consumer) == StyleStatus::InvalidArgument);

    // Empty geometry (including empty point-domain fields) is a successful no-op.
    DeviceBuffer<uint32_t> emptyOffsets; CHECK(emptyOffsets.reset(1) == cudaSuccess);
    CHECK(Upload(emptyOffsets, {0u}, producer) && cudaStreamSynchronize(producer) == cudaSuccess);
    DeviceCurveGeometryView empty{{}, {}, {}, {emptyOffsets.data(),1}, {}, 0, 0};
    CudaLength emptyOp;
    CHECK(emptyOp.Apply(empty, {}, LengthParameters{}, {}, {}, producer) == StyleStatus::Ok &&
          emptyOp.Finish(consumer) == StyleStatus::Ok && emptyOp.deviceIndex() >= 0);

    // Fresh Length conditionally publishes both candidate points and keep only
    // after its device validation scalar is proved by the relay callback.
    CHECK(Upload(points, source, producer) && Upload(offsets, {0u,3u,5u}, producer));
    CudaLength fresh;
    FreshSignal freshSignal;
    p=LengthParameters{}; p.value=ScalarField::Literal(2.f);
    CHECK(fresh.ApplyFresh(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::Ok &&
          fresh.HasUnprovenWork() && fresh.CommitFreshFinish() == StyleStatus::InvalidArgument &&
          fresh.Finish(producer) == StyleStatus::InvalidArgument &&
          fresh.Apply(geometry, {}, p, output.view(), keep.view(), producer) == StyleStatus::InvalidArgument &&
          fresh.FinishFreshAsync(producer, FreshCallback, &freshSignal) == StyleStatus::Ok &&
          cudaStreamSynchronize(producer) == cudaSuccess && freshSignal.calls.load(std::memory_order_acquire)==1 &&
          freshSignal.status.load(std::memory_order_acquire)==int(cudaSuccess) &&
          fresh.CommitFreshFinish() == StyleStatus::Ok && !fresh.HasUnprovenWork() &&
          Download(output,got,consumer) && Download(keep,gotKeep,consumer));
    CHECK(Near(got[1].x,2) && Near(got[2].y,2) && Near(got[4].z,6) &&
          gotKeep==std::vector<uint8_t>({1,1}));

    auto freshSemanticFailure = [&](LengthParameters const& invalid) {
        std::vector<unsigned char> pointSentinel(sizeof(float3)*got.size(),0x6D);
        std::vector<uint8_t> keepSentinel(keep.size(),0xA7);
        if (cudaMemsetAsync(output.data(),0x6D,output.size()*sizeof(float3),producer)!=cudaSuccess ||
            cudaMemsetAsync(keep.data(),0xA7,keep.size(),producer)!=cudaSuccess) return false;
        FreshSignal signal;
        if (fresh.ApplyFresh(geometry,{},invalid,output.view(),keep.view(),producer)!=StyleStatus::Ok ||
            fresh.FinishFreshAsync(producer,FreshCallback,&signal)!=StyleStatus::Ok ||
            cudaStreamSynchronize(producer)!=cudaSuccess || signal.calls.load(std::memory_order_acquire)!=1 ||
            signal.status.load(std::memory_order_acquire)!=int(cudaSuccess) ||
            fresh.CommitFreshFinish()==StyleStatus::Ok || fresh.HasUnprovenWork() ||
            !Download(output,got,consumer) || !Download(keep,gotKeep,consumer)) return false;
        return std::memcmp(got.data(),pointSentinel.data(),pointSentinel.size())==0 && gotKeep==keepSentinel;
    };
    // Invalid offsets, a device profile value, and an expression-like device
    // point field are all semantic failures: neither candidate channel leaks.
    CHECK(Upload(offsets,{0u,3u,9u},producer) && freshSemanticFailure(LengthParameters{}));
    CHECK(Upload(offsets,{0u,3u,5u},producer));
    DeviceBuffer<float> badMask, badExpression;
    CHECK(badMask.reset(5)==cudaSuccess && badExpression.reset(5)==cudaSuccess &&
          Upload(badMask,{1.f,NAN,1.f,1.f,1.f},producer) &&
          Upload(badExpression,{1.f,-1.f,1.f,1.f,1.f},producer));
    p=LengthParameters{}; p.mask=ScalarField::Device(Const(badMask),expr::Domain::Point);
    CHECK(freshSemanticFailure(p));
    p=LengthParameters{}; p.value=ScalarField::Device(Const(badExpression),expr::Domain::Point);
    CHECK(freshSemanticFailure(p));
    // A proven semantic rejection leaves this instance reusable.
    FreshSignal freshRetry;
    CHECK(fresh.ApplyFresh(geometry,{},LengthParameters{},output.view(),keep.view(),producer)==StyleStatus::Ok &&
          fresh.FinishFreshAsync(producer,FreshCallback,&freshRetry)==StyleStatus::Ok &&
          cudaStreamSynchronize(producer)==cudaSuccess && freshRetry.calls.load()==1 &&
          fresh.CommitFreshFinish()==StyleStatus::Ok);

    // Capture is refused before the fresh path can query/allocate/submit.
    CudaLength capture;
    cudaGraph_t graph=nullptr;
    CHECK(cudaStreamBeginCapture(producer,cudaStreamCaptureModeGlobal)==cudaSuccess &&
          capture.ApplyFresh(geometry,{},LengthParameters{},output.view(),keep.view(),producer)==StyleStatus::InvalidArgument &&
          cudaStreamEndCapture(producer,&graph)==cudaSuccess);
    if (graph) CHECK(cudaGraphDestroy(graph)==cudaSuccess);
    CudaLength freshEmpty;
    FreshSignal emptySignal;
    CHECK(freshEmpty.ApplyFresh(empty,{},LengthParameters{},{},{},producer)==StyleStatus::Ok &&
          freshEmpty.FinishFreshAsync(producer,FreshCallback,&emptySignal)==StyleStatus::Ok &&
          cudaStreamSynchronize(producer)==cudaSuccess && emptySignal.calls.load()==1 &&
          freshEmpty.CommitFreshFinish()==StyleStatus::Ok);

    CHECK(cudaStreamDestroy(producer) == cudaSuccess && cudaStreamDestroy(consumer) == cudaSuccess);
    std::puts("testUsdGenCudaLength: PASS");
    return 0;
}
