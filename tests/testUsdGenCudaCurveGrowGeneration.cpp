#include "usdGen/gpu/curveGrow.h"
#include "usdGen/gpu/generation.h"
#include "usdGen/gpu/deviceResources.h"
#include "usdGen/executionRetirement.h"

#include <atomic>
#include <cstdio>
#include <memory>
#include <vector>

using namespace usdGen;
using namespace usdGen::gpu;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #c); return 1; } } while (false)

namespace {
template<class T> bool Upload(DeviceBuffer<T>& buffer, std::vector<T> const& values) {
    return buffer.reset(values.size()) == cudaSuccess &&
        cudaMemcpy(buffer.data(), values.data(), values.size() * sizeof(T),
                   cudaMemcpyHostToDevice) == cudaSuccess;
}
template<class T> bool Read(DeviceView<const T> view, std::vector<T>* values,
                            cudaStream_t stream) {
    values->resize(view.size);
    return cudaMemcpyAsync(values->data(), view.data, view.size * sizeof(T),
                           cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
        cudaStreamSynchronize(stream) == cudaSuccess;
}
struct Input {
    DeviceBuffer<float3> points, rest, t, b, n;
    DeviceBuffer<float> widths;
    DeviceBuffer<uint32_t> offsets;
    DeviceBuffer<uint64_t> ids;
    DeviceBuffer<int32_t> prim;
    DeviceBuffer<float2> uv;
    CurveGrowInput View() const {
        return {{points.view(), rest.view(), widths.view(), offsets.view(), ids.view(), 1, 2},
                prim.view(), uv.view(), t.view(), b.view(), n.view()};
    }
};
void Done(cudaStream_t, cudaError_t status, void* data) noexcept {
    static_cast<std::atomic<int>*>(data)->store(int(status));
}
std::unique_ptr<CudaCurveGrow> CompleteGrow(CurveGrowInput input,
                                            std::shared_ptr<const void> owner,
                                            cudaStream_t stream,
                                            CurveGrowControls controls = {}) {
    auto grow=std::make_unique<CudaCurveGrow>();
    std::atomic<int> completion{-1};
    if(grow->BeginFresh(input,std::move(owner),controls,stream)!=CurveGrowStatus::Ok ||
       grow->FinishFreshAsync(stream,Done,&completion)!=CurveGrowStatus::Ok ||
       cudaStreamSynchronize(stream)!=cudaSuccess || completion.load()!=int(cudaSuccess) ||
       grow->CommitFreshFinish()!=CurveGrowStatus::Ok)
        return {};
    return grow;
}
}

int main() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return 77;
    CHECK(ConfigureCudaExecutionResources(device, {size_t{8} << 20, 0}));
    auto key = UsdGenExecutionResourceDevice{UsdGenExecutionResourceBackend::Cuda, device};
    auto pool = FindUsdGenExecutionResourcePool(key);
    auto retirement = GetOrCreateUsdGenExecutionRetirementService(key, {64});
    CHECK(pool && retirement);
    auto const baseline = pool->Snapshot().usedBytes;
    cudaStream_t producer = nullptr, consumer = nullptr;
    CHECK(cudaStreamCreate(&producer) == cudaSuccess);
    CHECK(cudaStreamCreate(&consumer) == cudaSuccess);
    {
        std::string reason;
        CHECK(!MakeCurveGrowGeneration({}, 1, &reason) && !reason.empty());
        CHECK(!MakeCurveGrowGeneration(std::make_unique<CudaCurveGrow>(), 1, &reason));
        auto input = std::make_shared<Input>();
        CHECK(Upload(input->points, {make_float3(10,0,0), make_float3(11,0,0)}));
        CHECK(Upload(input->rest, {make_float3(1,0,0), make_float3(2,0,0)}));
        CHECK(Upload(input->widths, {.2f, .4f}));
        CHECK(Upload(input->offsets, {0u,2u}) && Upload(input->ids, {91ull}));
        CHECK(Upload(input->prim, {7}) && Upload(input->uv, {make_float2(.25f,.5f)}));
        CHECK(Upload(input->t, {make_float3(1,0,0)}));
        CHECK(Upload(input->b, {make_float3(0,1,0)}));
        CHECK(Upload(input->n, {make_float3(0,0,1)}));
        auto grow = std::make_unique<CudaCurveGrow>();
        CurveGrowControls controls; controls.cvCount = 3; controls.length = 2;
        CHECK(grow->BeginFresh(input->View(), input, controls, producer) == CurveGrowStatus::Ok);
        std::atomic<int> completion{-1};
        CHECK(grow->FinishFreshAsync(producer, Done, &completion) == CurveGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(producer) == cudaSuccess && completion.load() == int(cudaSuccess));
        CHECK(grow->CommitFreshFinish() == CurveGrowStatus::Ok);
        std::vector<float3> unchanged;
        CHECK(Read(input->View().geometry.points, &unchanged, producer));
        CHECK(unchanged[0].x == 10 && unchanged[1].x == 11 && unchanged[1].z == 0);
        input.reset();

        CudaNamedChannelPlane plane;
        plane.metadata = {"density", UsdGenDeviceValueType::Float32,
            UsdGenDeviceDomain::Primitive, 1, 1, sizeof(float), true,
            UsdGenDeviceChannelSemantic::Generic};
        plane.bytes = std::make_unique<DeviceBuffer<unsigned char>>();
        float density = .75f;
        CHECK(plane.bytes->reset(sizeof(float)) == cudaSuccess);
        CHECK(cudaMemcpy(plane.bytes->data(), &density, sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
        std::vector<CudaNamedChannelPlane> planes;
        planes.push_back(std::move(plane));
        auto generation = MakeCurveGrowGeneration(std::move(grow), 10, &reason,
                                                   false, 4, std::move(planes));
        if (!generation) std::fprintf(stderr, "%s\n", reason.c_str());
        CHECK(generation && generation->Owner()->ProducerReady());
        CHECK(generation->Identity().generation == 10 && generation->Geometry().topologyVersion == 4);
        auto old = AcquireGeometry(generation, consumer);
        CHECK(old && old.Geometry().pointCount == 3 && old.Geometry().curveCount == 1);
        std::vector<float3> points, rest, frame;
        std::vector<float> widths, hair;
        std::vector<uint32_t> offsets;
        std::vector<uint64_t> ids;
        std::vector<int32_t> prim;
        std::vector<float2> uv;
        CHECK(Read(old.Geometry().points, &points, consumer) && Read(old.Geometry().restPoints, &rest, consumer));
        CHECK(points[0].x == 10 && points[2].z == 2 && rest[0].x == 1 && rest[2].z == 2);
        CHECK(Read(old.Geometry().widths, &widths, consumer) && widths.front() == .2f && widths.back() == .4f);
        CHECK(Read(old.HairT(), &hair, consumer) && hair == std::vector<float>({0,.5f,1}));
        CHECK(Read(old.Geometry().curveOffsets, &offsets, consumer) && offsets == std::vector<uint32_t>({0,3}));
        CHECK(Read(old.Geometry().stableIds, &ids, consumer) && ids == std::vector<uint64_t>({91}));
        CHECK(Read(old.RootPrim(), &prim, consumer) && prim == std::vector<int32_t>({7}));
        CHECK(Read(old.RootUV(), &uv, consumer) && uv[0].x == .25f && uv[0].y == .5f);
        CHECK(Read(old.RootT(), &frame, consumer) && frame[0].x == 1);
        CHECK(Read(old.RootB(), &frame, consumer) && frame[0].y == 1);
        CHECK(Read(old.RootN(), &frame, consumer) && frame[0].z == 1);

        auto editedPoints = std::make_unique<DeviceBuffer<float3>>();
        auto edited = points;
        edited[2].x = 30;
        CHECK(Upload(*editedPoints, edited));
        auto revision = MakePointRevisionGeneration(generation, 11, std::move(editedPoints), {}, &reason);
        CHECK(revision && revision->Geometry().topologyVersion == 4);
        auto latest = AcquireGeometry(revision, consumer);
        CHECK(latest && latest.Geometry().points.data != old.Geometry().points.data);
        CHECK(latest.Geometry().restPoints.data == old.Geometry().restPoints.data);
        CHECK(latest.Geometry().widths.data == old.Geometry().widths.data);
        CHECK(latest.RootB().data == old.RootB().data);
        auto named = AcquireNamedChannel(revision, "density", consumer);
        CHECK(named && named.Bytes().size == sizeof(float));
        generation.reset(); revision.reset();
        // Both generations are now held solely by consumer leases. The old
        // points remain unchanged and the generic plane remains readable.
        CHECK(Read(old.Geometry().points, &points, consumer) && points[2].x == 10);
        CHECK(Read(latest.Geometry().points, &points, consumer) && points[2].x == 30);
        float readDensity = 0;
        CHECK(cudaMemcpyAsync(&readDensity, named.Bytes().data, sizeof(float), cudaMemcpyDeviceToHost, consumer) == cudaSuccess);
        CHECK(cudaStreamSynchronize(consumer) == cudaSuccess && readDensity == density);
    }
    retirement->Drain();
    CHECK(pool->Snapshot().usedBytes == baseline);
    {
        // Factory-local point/width revisions are fresh COW values.  The
        // retained CurveGrow owner must continue to provide rest, IDs,
        // bindings, and root frames after the caller drops its input owner.
        auto input=std::make_shared<Input>();
        CHECK(Upload(input->points,{make_float3(4,5,6),make_float3(4,5,7)}));
        CHECK(Upload(input->rest,{make_float3(1,2,3),make_float3(1,2,4)}));
        CHECK(Upload(input->widths,{.1f,.2f})&&Upload(input->offsets,{0u,2u})&&
              Upload(input->ids,{31ull})&&Upload(input->prim,{9})&&
              Upload(input->uv,{make_float2(.1f,.2f)})&&
              Upload(input->t,{make_float3(1,0,0)})&&
              Upload(input->b,{make_float3(0,1,0)})&&
              Upload(input->n,{make_float3(0,0,1)}));
        CurveGrowControls controls; controls.cvCount=3;
        auto grow=CompleteGrow(input->View(),input,producer,controls);
        CHECK(grow);
        auto const baseView=grow->view();
        auto const baseT=grow->rootT(), baseB=grow->rootB(), baseN=grow->rootN();
        auto replacementWidths=std::make_unique<DeviceBuffer<float>>();
        auto replacementPoints=std::make_unique<DeviceBuffer<float3>>();
        CHECK(Upload(*replacementWidths,{.7f,.8f,.9f}));
        CHECK(Upload(*replacementPoints,{make_float3(20,0,0),make_float3(21,0,0),make_float3(22,0,0)}));
        auto const* widthPtr=replacementWidths->data();
        auto const* pointPtr=replacementPoints->data();
        input.reset();
        std::string reason;
        auto generation=MakeCurveGrowGeneration(std::move(grow),20,&reason,false,8,{},
            std::move(replacementWidths),std::move(replacementPoints));
        CHECK(generation&&generation->Owner()->ProducerReady());
        auto lease=AcquireGeometry(generation,consumer);
        CHECK(lease&&lease.Geometry().points.data==pointPtr&&lease.Geometry().widths.data==widthPtr);
        CHECK(lease.Geometry().restPoints.data==baseView.restPoints.data&&
              lease.Geometry().stableIds.data==baseView.stableIds.data&&
              lease.RootT().data==baseT.data&&lease.RootB().data==baseB.data&&lease.RootN().data==baseN.data);
        std::vector<float> widths; std::vector<float3> points;
        CHECK(Read(lease.Geometry().widths,&widths,consumer)&&widths==std::vector<float>({.7f,.8f,.9f}));
        CHECK(Read(lease.Geometry().points,&points,consumer)&&points[2].x==22);
    }
    retirement->Drain();
    CHECK(pool->Snapshot().usedBytes == baseline);
    {
        // Safe wrong-shape device allocations must fail factory admission
        // before publication; completed owners are then normally destroyed.
        auto input=std::make_shared<Input>();
        CHECK(Upload(input->points,{make_float3(0,0,0),make_float3(0,0,1)})&&
              Upload(input->rest,{make_float3(0,0,0),make_float3(0,0,1)})&&
              Upload(input->widths,{.1f,.2f})&&Upload(input->offsets,{0u,2u})&&
              Upload(input->ids,{4ull})&&Upload(input->prim,{0})&&
              Upload(input->uv,{make_float2(0,0)})&&Upload(input->t,{make_float3(1,0,0)})&&
              Upload(input->b,{make_float3(0,1,0)})&&Upload(input->n,{make_float3(0,0,1)}));
        auto badWidths=std::make_unique<DeviceBuffer<float>>(); CHECK(Upload(*badWidths,{.1f}));
        std::string reason;
        auto widthCandidate = CompleteGrow(input->View(),input,producer);
        CHECK(widthCandidate);
        CHECK(!MakeCurveGrowGeneration(std::move(widthCandidate),21,&reason,
              false,UINT64_MAX,{},std::move(badWidths)) &&
              reason.find("width revision has incompatible shape") != std::string::npos);
        auto badPoints=std::make_unique<DeviceBuffer<float3>>(); CHECK(Upload(*badPoints,{make_float3(0,0,0)}));
        reason.clear();
        auto pointCandidate = CompleteGrow(input->View(),input,producer);
        CHECK(pointCandidate);
        CHECK(!MakeCurveGrowGeneration(std::move(pointCandidate),22,&reason,
              false,UINT64_MAX,{}, {},std::move(badPoints)) &&
              reason.find("point revision has incompatible shape") != std::string::npos);
    }
    retirement->Drain();
    CHECK(pool->Snapshot().usedBytes == baseline);
    {
        // Empty topology still publishes a real offsets owner; it must not
        // require non-null zero-length root planes or invent one curve.
        auto input = std::make_shared<Input>();
        CHECK(Upload(input->offsets, {0u}));
        auto view = input->View();
        view.geometry.curveCount = 0;
        view.geometry.pointCount = 0;
        auto grow = std::make_unique<CudaCurveGrow>();
        std::atomic<int> completion{-1};
        CHECK(grow->BeginFresh(view, input, {}, producer) == CurveGrowStatus::Ok);
        CHECK(grow->FinishFreshAsync(producer, Done, &completion) == CurveGrowStatus::Ok);
        CHECK(cudaStreamSynchronize(producer) == cudaSuccess && completion.load() == int(cudaSuccess));
        CHECK(grow->CommitFreshFinish() == CurveGrowStatus::Ok);
        input.reset();
        auto generation = MakeCurveGrowGeneration(std::move(grow), 12);
        CHECK(generation && generation->Geometry().elementCount == 0 && generation->Geometry().pointCount == 0);
        auto lease = AcquireGeometry(generation, consumer);
        CHECK(lease && lease.RootT().size == 0 && lease.RootPrim().size == 0);
        std::vector<uint32_t> offsets;
        CHECK(Read(lease.Geometry().curveOffsets, &offsets, consumer) && offsets == std::vector<uint32_t>({0}));
    }
    retirement->Drain();
    CHECK(pool->Snapshot().usedBytes == baseline);
    CHECK(cudaStreamDestroy(consumer) == cudaSuccess);
    CHECK(cudaStreamDestroy(producer) == cudaSuccess);
    return 0;
}
