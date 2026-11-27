#include "usdGenCurves/cuda/deformCurves.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace usdGen::gpu;
#define CHECK(X) do { if (!(X)) { \
    std::fprintf(stderr, "check failed at %d: %s\n", __LINE__, #X); return 1; \
} } while (0)

static float3 P(float x, float y, float z) { return make_float3(x, y, z); }
static bool Near(float3 a, float3 b) {
    return std::fabs(a.x-b.x) < 4.e-4f && std::fabs(a.y-b.y) < 4.e-4f &&
           std::fabs(a.z-b.z) < 4.e-4f;
}
template<class T>
static bool Upload(DeviceBuffer<T>& buffer, std::vector<T> const& values) {
    return buffer.reset(values.size()) == cudaSuccess &&
        cudaMemcpy(buffer.data(), values.data(), values.size()*sizeof(T),
                   cudaMemcpyHostToDevice) == cudaSuccess;
}
template<class T>
static std::vector<T> Read(DeviceBuffer<T> const& buffer) {
    std::vector<T> values(buffer.size());
    if (cudaMemcpy(values.data(), buffer.data(), values.size()*sizeof(T),
                   cudaMemcpyDeviceToHost) != cudaSuccess) values.clear();
    return values;
}

int main() {
    cudaStream_t producer = nullptr, consumer = nullptr;
    const cudaError_t initialized = cudaStreamCreate(&producer);
    if (initialized != cudaSuccess) {
        std::fprintf(stderr, "CUDA initialization: %s\n", cudaGetErrorString(initialized));
        return 1;
    }
    CHECK(cudaStreamCreate(&consumer) == cudaSuccess);
    {
        const std::vector<float3> samples = {
            P(0,0,0), P(1,0,0), P(0,1,0), P(0,0,1), P(1,1,1)};
        const std::vector<float3> rest = {
            P(.2,.1,.1), P(.3,.1,.4), P(.6,.2,.1), P(.7,.4,.3), P(.5,.6,.5)};
        const std::vector<uint32_t> offsets = {0,2,5}; // variable 2/3 CVs
        DeviceBuffer<float3> drivers, posed, points, targets, output, warp;
        DeviceBuffer<uint32_t> deviceOffsets;
        DeviceBuffer<float> primitive, pointMask, blend, mask;
        DeviceBuffer<float3> incoming;
        DeviceBuffer<uint8_t> enabled, lockRoots;
        CHECK(Upload(drivers, samples) && Upload(points, rest) &&
              Upload(deviceOffsets, offsets));
        CHECK(output.reset(5) == cudaSuccess && warp.reset(5) == cudaSuccess);
        DeviceCurveGeometryView geometry = {
            {points.data(),5}, {points.data(),5}, {}, {deviceOffsets.data(),3}, {}, 2,5};
        CudaRbfBinding binding;
        CudaRbfCurveDeformer deform;
        CHECK(binding.Bind({drivers.data(),5}, 0, producer) == RbfStatus::Ok);

        // Affine motion is exactly representable; roots follow explicit targets.
        auto animated = samples;
        for (auto& p : animated) p = P(-p.y+2, p.x-3, p.z+4);
        CHECK(Upload(posed, animated));
        CHECK(Upload(targets, std::vector<float3>{
            P(-rest[0].y+2,rest[0].x-3,rest[0].z+4),
            P(-rest[2].y+2,rest[2].x-3,rest[2].z+4)}));
        CHECK(binding.Solve({posed.data(),5}, producer) == RbfStatus::Ok);
        CHECK(deform.Deform(binding, geometry, {targets.data(),2}, 1, {}, {},
                            output.view(), producer) == RbfStatus::Ok);
        CHECK(deform.Finish(binding, consumer) == RbfStatus::Ok);
        auto actual = Read(output);
        CHECK(actual.size() == 5);
        for (unsigned i=0; i<5; ++i)
            CHECK(Near(actual[i], P(-rest[i].y+2,rest[i].x-3,rest[i].z+4)));

        // Reuse the rest binding for nonlinear motion. Check the whole curve's
        // root correction, not just whether some point changed.
        animated = samples;
        for (auto& p : animated) p.z += p.x*p.y;
        CHECK(Upload(posed, animated));
        CHECK(binding.Solve({posed.data(),5}, producer) == RbfStatus::Ok);
        CHECK(binding.Evaluate({points.data(),5}, warp.view(), producer) == RbfStatus::Ok);
        CHECK(binding.Finish(consumer) == RbfStatus::Ok);
        const auto warped = Read(warp);
        CHECK(warped.size() == 5 && !Near(warped[3], rest[3]));
        CHECK(Upload(targets, std::vector<float3>{rest[0],rest[2]}));
        CHECK(deform.Deform(binding, geometry, {targets.data(),2}, 1, {}, {},
                            output.view(), producer) == RbfStatus::Ok);
        CHECK(deform.Finish(binding, consumer) == RbfStatus::Ok);
        actual = Read(output);
        CHECK(actual.size() == 5);
        std::vector<float3> locked(5);
        for (unsigned i=0; i<5; ++i) {
            const unsigned root = i < 2 ? 0 : 2;
            locked[i] = P(warped[i].x + rest[root].x - warped[root].x,
                          warped[i].y + rest[root].y - warped[root].y,
                          warped[i].z + rest[root].z - warped[root].z);
            CHECK(Near(actual[i], locked[i]));
        }
        CHECK(Near(actual[0],rest[0]) && Near(actual[2],rest[2]));
        CHECK(!Near(actual[3],rest[3]));

        // Every envelope domain composes; zero, half and full remain numeric
        // oracles with variable CV counts and a different completion stream.
        const std::vector<float> primWeights = {0.5f,1.f};
        const std::vector<float> pointWeights = {1.f,0.f,1.f,0.5f,1.f};
        CHECK(Upload(primitive, primWeights) && Upload(pointMask, pointWeights));
        for (float groom : {0.f,0.5f,1.f}) {
            CHECK(deform.Deform(binding, geometry, {targets.data(),2}, groom,
                {primitive.data(),2}, {pointMask.data(),5}, output.view(), producer) == RbfStatus::Ok);
            CHECK(deform.Finish(binding, consumer) == RbfStatus::Ok);
            actual = Read(output);
            CHECK(actual.size() == 5);
            for (unsigned i=0; i<5; ++i) {
                const float e = groom * primWeights[i<2 ? 0 : 1] * pointWeights[i];
                CHECK(Near(actual[i], P(rest[i].x+(locked[i].x-rest[i].x)*e,
                    rest[i].y+(locked[i].y-rest[i].y)*e,
                    rest[i].z+(locked[i].z-rest[i].z)*e)));
            }
        }

        // The typed path consumes incoming styled points, not canonical
        // restPoints.  Exercise all supported field domains, including a
        // primitive-domain whole-strand lock.  usdGen:mask is the whole
        // envelope: the former along-strand profile is now just an expression.
        const std::vector<float3> styled = {
            P(3,.2f,-1), P(4,.3f,-2), P(5,.4f,-3), P(6,.5f,-4), P(7,.6f,-5)};
        const std::vector<float> blendValues = {.5f, 1.f};
        // Point-domain envelope; the zero entry keeps a locked-root CV an
        // exact pass-through, as the old zero ramp sample did.
        const std::vector<float> maskValues = {1.f,.5f,1.f,0.f,1.f};
        CHECK(Upload(incoming, styled) && Upload(blend, blendValues) &&
              Upload(mask, maskValues) &&
              Upload(enabled, std::vector<uint8_t>{1u}) &&
              Upload(lockRoots, std::vector<uint8_t>{0u,1u}));
        DeviceCurveGeometryView styledGeometry = geometry;
        styledGeometry.points = {incoming.data(), 5};
        DeformParameters parameters;
        parameters.mask = ScalarField::Device({mask.data(),5}, usdGen::expr::Domain::Point);
        parameters.enabled = BoolField::Device({enabled.data(),1}, usdGen::expr::Domain::Groom);
        parameters.lockRoots = BoolField::Device({lockRoots.data(),2}, usdGen::expr::Domain::Primitive);
        CHECK(binding.Evaluate({incoming.data(),5}, warp.view(), producer) == RbfStatus::Ok);
        CHECK(binding.Finish(consumer) == RbfStatus::Ok);
        const auto typedWarped = Read(warp);
        CHECK(typedWarped.size() == 5);
        CHECK(deform.Deform(binding, styledGeometry, {targets.data(),2}, parameters,
                            output.view(), producer) == RbfStatus::Ok);
        CHECK(deform.Finish(binding, consumer) == RbfStatus::Ok);
        actual = Read(output);
        CHECK(actual.size() == styled.size());
        for (unsigned i = 0; i < actual.size(); ++i) {
            const unsigned curve = i < 2 ? 0 : 1;
            const unsigned root = curve == 0 ? 0 : 2;
            const float e = maskValues[i];
            const float3 destination = curve == 1
                ? P(typedWarped[i].x + rest[root].x - typedWarped[root].x,
                    typedWarped[i].y + rest[root].y - typedWarped[root].y,
                    typedWarped[i].z + rest[root].z - typedWarped[root].z)
                : typedWarped[i];
            CHECK(Near(actual[i], P(styled[i].x + (destination.x-styled[i].x)*e,
                styled[i].y + (destination.y-styled[i].y)*e,
                styled[i].z + (destination.z-styled[i].z)*e)));
        }
        // A zero mask is also a strict incoming-geometry pass-through; this
        // catches accidentally using canonical restPoints as the source.
        parameters.mask = ScalarField::Literal(0.f);
        parameters.enabled = BoolField::Literal(true);
        CHECK(deform.Deform(binding, styledGeometry, {targets.data(),2}, parameters,
                            output.view(), producer) == RbfStatus::Ok);
        CHECK(deform.Finish(binding, consumer) == RbfStatus::Ok);
        actual = Read(output);
        for (unsigned i = 0; i < actual.size(); ++i) CHECK(Near(actual[i], styled[i]));
        // A disabled groom still validates its controls, but is a strict
        // incoming-geometry pass-through and never publishes canonical rest.
        parameters.enabled = BoolField::Literal(false);
        CHECK(deform.Deform(binding, styledGeometry, {targets.data(),2}, parameters,
                            output.view(), producer) == RbfStatus::Ok);
        CHECK(deform.Finish(binding, consumer) == RbfStatus::Ok);
        actual = Read(output);
        for (unsigned i = 0; i < actual.size(); ++i) CHECK(Near(actual[i], styled[i]));
        // Fresh binding uses independent host-only proof commits.  The
        // stream synchronizations below are test-only stand-ins for the
        // production native terminal callbacks.
        CudaRbfBinding freshBinding;
        CHECK(freshBinding.BeginFreshBind({drivers.data(),5},0,producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && freshBinding.CommitFreshBindExtent()==RbfStatus::Ok);
        CHECK(freshBinding.BeginFreshBindRank(producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && freshBinding.CommitFreshBindRank()==RbfStatus::Ok);
        CHECK(freshBinding.BeginFreshBindLu(producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && freshBinding.CommitFreshBindLu()==RbfStatus::Ok);
        parameters.enabled=BoolField::Device({enabled.data(),1},usdGen::expr::Domain::Groom);
        parameters.mask=ScalarField::Device({mask.data(),5},usdGen::expr::Domain::Point);
        const auto freshSentinel=Read(output); CudaRbfCurveDeformer fresh;
        CHECK(fresh.BeginFreshShape(styledGeometry,{targets.data(),2},parameters,output.view(),producer)==RbfStatus::Ok);
        CHECK(fresh.HasUnprovenWork());
        CHECK(fresh.Deform(binding,styledGeometry,{targets.data(),2},parameters,output.view(),producer)==RbfStatus::InvalidArgument);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshShape()==RbfStatus::Ok);
        CHECK(fresh.BeginFreshEvaluate(freshBinding,producer)==RbfStatus::Ok);
        CHECK(fresh.CommitFreshEvaluate(binding)==RbfStatus::InvalidArgument && fresh.HasUnprovenWork());
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshEvaluate(freshBinding)==RbfStatus::Ok);
        CHECK(fresh.BeginFreshApply(producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshApply()==RbfStatus::Ok);
        actual=Read(output); for(unsigned i=0;i<actual.size();++i) CHECK(Near(actual[i],freshSentinel[i]));
        CHECK(fresh.BeginFreshCopy(producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshFinish()==RbfStatus::Ok);
        // Accepted fresh binding is identity: primitive 0 is unlocked and
        // remains styled, while primitive 1's locked root reaches rest[2].
        actual=Read(output); CHECK(actual.size()==styled.size());
        CHECK(Near(actual[0],styled[0]) && Near(actual[2],rest[2]) && Near(actual[3],styled[3]));
        // Malformed offsets fail at the shape proof and do not touch output;
        // the same fresh object can start another proven candidate afterward.
        const auto beforeFreshFailure=Read(output);
        float3 hostPoint{}; DeviceCurveGeometryView hostPointerGeometry=styledGeometry;
        hostPointerGeometry.points={&hostPoint,5};
        CHECK(fresh.BeginFreshShape(hostPointerGeometry,{targets.data(),2},parameters,output.view(),producer)==RbfStatus::InvalidArgument);
        CHECK(Upload(deviceOffsets,std::vector<uint32_t>{0,2,6})); styledGeometry.curveOffsets={deviceOffsets.data(),3};
        CHECK(fresh.BeginFreshShape(styledGeometry,{targets.data(),2},parameters,output.view(),producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshShape()==RbfStatus::InvalidArgument);
        actual=Read(output); for(unsigned i=0;i<actual.size();++i) CHECK(Near(actual[i],beforeFreshFailure[i]));
        CHECK(Upload(deviceOffsets,offsets)); styledGeometry.curveOffsets={deviceOffsets.data(),3};
        // The same object admits a complete retry after a proven semantic
        // rejection; no failed phase remains sticky.
        CHECK(fresh.BeginFreshShape(styledGeometry,{targets.data(),2},parameters,output.view(),producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshShape()==RbfStatus::Ok);
        CHECK(fresh.BeginFreshEvaluate(freshBinding,producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshEvaluate(freshBinding)==RbfStatus::Ok);
        CHECK(fresh.BeginFreshApply(producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshApply()==RbfStatus::Ok);
        CHECK(fresh.BeginFreshCopy(producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshFinish()==RbfStatus::Ok);
        // Device-field NaN is an Apply semantic rejection and never copies
        // staging into the caller's sentinel destination.
        const auto beforeFreshNaN=Read(output);
        CHECK(Upload(mask,std::vector<float>{NAN,.5f,1.f,.5f,1.f}));
        CHECK(fresh.BeginFreshShape(styledGeometry,{targets.data(),2},parameters,output.view(),producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshShape()==RbfStatus::Ok);
        CHECK(fresh.BeginFreshEvaluate(freshBinding,producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshEvaluate(freshBinding)==RbfStatus::Ok);
        CHECK(fresh.BeginFreshApply(producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && fresh.CommitFreshApply()==RbfStatus::NonFiniteInput);
        actual=Read(output); for(unsigned i=0;i<actual.size();++i) CHECK(Near(actual[i],beforeFreshNaN[i]));
        CHECK(Upload(mask,maskValues));
        DeviceBuffer<uint32_t> emptyOffsets;
        CHECK(Upload(emptyOffsets, std::vector<uint32_t>{0}));
        DeviceCurveGeometryView emptyGeometry{};
        emptyGeometry.curveOffsets = {emptyOffsets.data(), 1};
        CudaRbfCurveDeformer freshEmpty;
        CHECK(freshEmpty.BeginFreshShape(emptyGeometry,{},DeformParameters{}, {},producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && freshEmpty.CommitFreshShape()==RbfStatus::Ok);
        // Empty geometry intentionally skips RBF evaluation but still proves
        // Apply and the final publication boundary.
        CHECK(freshEmpty.BeginFreshEvaluate(freshBinding,producer)==RbfStatus::Ok);
        CHECK(freshEmpty.CommitFreshEvaluate(freshBinding)==RbfStatus::Ok);
        CHECK(freshEmpty.BeginFreshApply(producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && freshEmpty.CommitFreshApply()==RbfStatus::Ok);
        CHECK(freshEmpty.BeginFreshCopy(producer)==RbfStatus::Ok);
        CHECK(cudaStreamSynchronize(producer)==cudaSuccess && freshEmpty.CommitFreshFinish()==RbfStatus::Ok);
        cudaGraph_t captureGraph=nullptr;
        CHECK(cudaStreamBeginCapture(producer,cudaStreamCaptureModeGlobal)==cudaSuccess);
        CudaRbfCurveDeformer captureFresh;
        CHECK(captureFresh.BeginFreshShape(styledGeometry,{targets.data(),2},parameters,output.view(),producer)==RbfStatus::InvalidArgument);
        CHECK(cudaStreamEndCapture(producer,&captureGraph)==cudaSuccess);
        if (captureGraph) CHECK(cudaGraphDestroy(captureGraph)==cudaSuccess);
        CHECK(deform.Deform(binding, emptyGeometry, {}, DeformParameters{}, {}, producer) ==
              RbfStatus::Ok);
        CHECK(deform.Finish(binding, consumer) == RbfStatus::Ok);
        DeviceCurveGeometryView legacyEmpty{};
        CHECK(deform.Deform(binding, legacyEmpty, {}, 1.f, {}, {}, {}, producer) ==
              RbfStatus::Ok);
        CHECK(deform.Finish(binding, consumer) == RbfStatus::Ok);

        const auto beforeInvalid = actual;
        for (auto invalid : {std::vector<uint32_t>{0,2,4},
                             std::vector<uint32_t>{0,2,6},
                             std::vector<uint32_t>{0,0,5},
                             std::vector<uint32_t>{1,2,5}}) {
            CHECK(Upload(deviceOffsets, invalid));
            geometry.curveOffsets = {deviceOffsets.data(),3};
            CHECK(deform.Deform(binding, geometry, {targets.data(),2}, 1, {}, {},
                                output.view(), producer) == RbfStatus::InvalidArgument);
            actual = Read(output);
            CHECK(actual.size() == beforeInvalid.size());
            for (unsigned i=0; i<actual.size(); ++i) CHECK(Near(actual[i],beforeInvalid[i]));
        }
        CHECK(Upload(deviceOffsets, offsets));
        geometry.curveOffsets = {deviceOffsets.data(),3};
        CHECK(deform.Deform(binding, geometry, {targets.data(),2}, NAN, {}, {},
                            output.view(), producer) == RbfStatus::InvalidArgument);
        CHECK(deform.Deform(binding, geometry, {targets.data(),2}, 1, {nullptr,2}, {},
                            output.view(), producer) == RbfStatus::InvalidArgument);
        CHECK(deform.Deform(binding, geometry, {targets.data(),2}, 1, {}, {pointMask.data(),4},
                            output.view(), producer) == RbfStatus::InvalidArgument);

        // Nonfinite device fields invalidate the generation at Finish.
        CHECK(Upload(primitive, std::vector<float>{NAN,1.f}));
        CHECK(deform.Deform(binding, geometry, {targets.data(),2}, 1,
            {primitive.data(),2}, {}, output.view(), producer) == RbfStatus::Ok);
        CHECK(deform.Finish(binding, consumer) == RbfStatus::NonFiniteInput);
        actual = Read(output);
        CHECK(actual.size() == beforeInvalid.size());
        for (unsigned i = 0; i < actual.size(); ++i) CHECK(Near(actual[i], beforeInvalid[i]));
    }
    CHECK(cudaStreamDestroy(consumer) == cudaSuccess);
    CHECK(cudaStreamDestroy(producer) == cudaSuccess);
    std::puts("testUsdGenCudaDeform: PASS");
    return 0;
}
