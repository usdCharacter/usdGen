#include "usdGenCurves/publish/curvePublication.h"
#include "testBackend.h"
#include "usdGen/session.h"
#include "usdGenUsd/lower.h"
#include "usdGen/sessionCooker.h"
#include "usdGen/exec/sessionDeviceProvider.h"
#include "usdGen/registries/backend.h"
#include "usdGen/registries/backendRegistry.h"
#include "usdGen/executionRetirement.h"
#include "usdGen/executionResources.h"
#include "usdGenCurves/cuda/cudaExecution.h"
#include "usdGen/imagePayload.h"
#include "usdGenCurves/cuda/generation.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace usdGen;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); return 1; } } while (false)

static bool DrainCudaRetirement() {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    auto service = FindUsdGenExecutionRetirementService(
        {usdGenTestUtils::Backend("cuda"), device});
    if (!service) return false;
    service->Drain();
    return true;
}

// The registered CUDA backend's Session provider.
static std::shared_ptr<UsdGenSessionDeviceProvider> CudaProvider() {
    UsdGenBackendDef const* def =
        UsdGenGetBackend(usdGenTestUtils::Backend("cuda"));
    std::string reason;
    return def && def->createProvider ? def->createProvider({}, &reason) : nullptr;
}

// One cooker turn through the CUDA provider, waited for here (step 8.3 moved
// the cooker's CUDA execution behind the provider). The result returns on
// the provider's completion thread; the test thread only waits.
static UsdGenGenerationConstPtr CookCuda(UsdGenSessionCooker& cooker,
    std::shared_ptr<UsdGenSessionDeviceProvider> const& provider,
    std::shared_ptr<const UsdGenGraphSpec> desc, double frame,
    UsdGenGenerationConstPtr previous, UsdGenStats stats, int device) {
    // Leaked: the runtime's workers must not be joined at exit.
    static UsdGenExecutionRuntime* runtime = new UsdGenExecutionRuntime(4);
    struct Done {
        std::atomic<bool> ready{false};
        UsdGenGenerationConstPtr generation;
    };
    auto done = std::make_shared<Done>();
    cooker.CookDeviceAsync(*runtime, UsdGenExecutionPipeline::Cancellation{},
        std::move(desc), UsdGenContext::Interactive, true, {}, frame,
        UsdGenCommitReason::SetTime, std::move(previous), std::move(stats), false, 0,
        device, provider,
        [](UsdGenSessionCooker::DeviceResultHandler handler) {
            auto shared = std::make_shared<UsdGenSessionCooker::DeviceResultHandler>(
                std::move(handler));
            return UsdGenSessionDeviceReturn(
                [shared](std::shared_ptr<const UsdGenSessionDeviceResult> result) {
                    (*shared)(std::move(result));
                    return true;
                });
        },
        [done](UsdGenGenerationConstPtr generation, std::exception_ptr) {
            done->generation = std::move(generation);
            done->ready.store(true, std::memory_order_release);
        });
    while (!done->ready.load(std::memory_order_acquire))
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return done->generation;
}

static UsdGenGraphDesc SourceDesc() {
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom/Description");
    desc.executionBackend = usdGenTestUtils::Backend("cuda");
    desc.defaultWidth = .025f;
    UsdGenNodeDesc node;
    node.path = desc.description.AppendChild(TfToken("Ops")).AppendChild(TfToken("Source"));
    node.type = TfToken("UsdGenCurveSource");
    node.curves = {SdfPath("/Source")};
    node.surfaces = {SdfPath("/Scalp")};
    UsdGenSurfaceDesc scalp;
    scalp.path = SdfPath("/Scalp");
    scalp.faceVertexCounts = VtIntArray(9,3);
    scalp.faceVertexIndices = VtIntArray(27,0);
    scalp.restPoints = {GfVec3f(0),GfVec3f(1,0,0),GfVec3f(0,1,0)};
    scalp.points = scalp.restPoints;
    for (size_t i = 0; i < 27; ++i) scalp.faceVertexIndices[i] = static_cast<int>(i % 3);
    desc.surfaces.push_back(scalp);
    desc.terminal = node.path;
    desc.nodes.push_back(node);
    UsdGenCurveSetDesc curves;
    curves.path = SdfPath("/Source"); curves.role = UsdGenLane::Chunked;
    curves.curveRole = TfToken("hair");
    curves.curveVertexCounts = {2,3};
    curves.points = {GfVec3f(10,0,0),GfVec3f(11,0,0),GfVec3f(20,0,0),GfVec3f(21,0,0),GfVec3f(24,0,0)};
    curves.rest = curves.points;
    curves.curveId = {91,37};
    curves.skinPrim = {4,8}; curves.skinPrimUv = {GfVec2f(.1f,.2f),GfVec2f(.3f,.4f)};
    desc.curveSets.push_back(curves);
    return desc;
}

static expr::ValueShape ScalarShape(expr::ScalarType scalar) {
    return {scalar, 1, 1, 1, 1, false};
}

static void AddCudaExpression(UsdGenGraphDesc* desc, char const* path,
                              char const* source, bool boolean = false) {
    UsdGenExpressionDesc expression;
    expression.path = SdfPath(path);
    expression.source = source;
    expression.outputs.push_back({TfToken("result"),
                                  TfToken(boolean ? "bool" : "float"),
                                  ScalarShape(boolean ? expr::ScalarType::Bool
                                                      : expr::ScalarType::Float32)});
    desc->expressions.push_back(std::move(expression));
}

static void BindCudaExpression(UsdGenNodeDesc* node, char const* expression,
                               char const* destination, expr::Domain domain,
                               bool boolean, VtValue literal) {
    UsdGenExpressionBinding binding;
    binding.expression = SdfPath(expression);
    binding.destination = TfToken(destination);
    binding.nativeType = TfToken(boolean ? "bool" : "float");
    binding.destinationShape = ScalarShape(boolean ? expr::ScalarType::Bool
                                                   : expr::ScalarType::Float32);
    binding.domain = domain;
    binding.literal = std::move(literal);
    node->expressionBindings.push_back(std::move(binding));
}

static UsdGenGraphDesc WidthChainDesc() {
    auto desc = SourceDesc();
    desc.timeCodesPerSecond = 24.0;

    UsdGenNodeDesc first;
    first.path = desc.description.AppendChild(TfToken("Ops")).AppendChild(TfToken("WidthLiteral"));
    first.type = TfToken("UsdGenWidth");
    first.inputs.push_back(desc.nodes.front().path);
    first.params.push_back({TfToken("width"), VtValue(0.1f), false});

    UsdGenNodeDesc second;
    second.path = desc.description.AppendChild(TfToken("Ops")).AppendChild(TfToken("WidthExpr"));
    second.type = TfToken("UsdGenWidth");
    second.inputs.push_back(first.path);
    second.params.push_back({TfToken("width"), VtValue(0.5f), false});
    desc.terminal = second.path;
    desc.nodes.push_back(std::move(first));
    desc.nodes.push_back(std::move(second));

    // The point field deliberately uses both commit frame and seconds.  The
    // usdGen: prefix is part of the authored transport contract and is
    // normalized by the CUDA executor before it reaches the primitive API.
    AddCudaExpression(&desc, "/Groom/Description/Expressions/pointWidth",
                      "$frame * 0.1 + $time");
    AddCudaExpression(&desc, "/Groom/Description/Expressions/replaceFirst",
                      "$primIndex == 0", true);
    AddCudaExpression(&desc, "/Groom/Description/Expressions/enabled",
                      "$frame < 3", true);
    BindCudaExpression(&desc.nodes[2],
                       "/Groom/Description/Expressions/pointWidth",
                       "usdGen:width", expr::Domain::Point, false, VtValue(0.5f));
    BindCudaExpression(&desc.nodes[2],
                       "/Groom/Description/Expressions/replaceFirst",
                       "replace", expr::Domain::Primitive, true, VtValue(false));
    BindCudaExpression(&desc.nodes[2],
                       "/Groom/Description/Expressions/enabled",
                       "enabled", expr::Domain::Groom, true, VtValue(true));
    return desc;
}

static UsdGenGraphDesc ImageWidthDesc() {
    auto desc = SourceDesc();
    UsdGenMapDesc map;
    map.path = desc.description.AppendChild(TfToken("Maps"))
        .AppendChild(TfToken("WidthMask"));
    map.type = TfToken("UsdGenImageMap");
    map.textureGeneration = 1;
    map.imagePayload = UsdGenImagePayload::Create(
        1, 1, 1, std::vector<float>{.5f},
        UsdGenImageRowOrientation::BottomUp);
    desc.maps.push_back(std::move(map));
    UsdGenNodeDesc width;
    width.path = desc.description.AppendChild(TfToken("Ops"))
        .AppendChild(TfToken("ImageWidth"));
    width.type = TfToken("UsdGenWidth");
    width.inputs = {desc.nodes.front().path};
    width.params.push_back({TfToken("width"), VtValue(.8f), false});
    width.mapBindings.push_back({desc.maps.front().path,
        TfToken("usdGen:map")});
    desc.nodes.push_back(std::move(width));
    desc.terminal = desc.nodes.back().path;
    return desc;
}

static UsdGenGraphDesc ReferenceSourceWidthDesc()
{
    UsdGenGraphDesc desc;
    desc.description = SdfPath("/Groom/ReferenceSource");
    desc.executionBackend = usdGenTestUtils::Backend("cuda");
    desc.defaultWidth = .025f;

    UsdGenCurveSetDesc reference;
    reference.path = SdfPath("/Reference/Guides");
    reference.role = UsdGenLane::Reference;
    reference.curveRole = TfToken("guide");
    reference.curveGeneration = 41;
    reference.curveVertexCounts = {2, 2};
    reference.points = {
        GfVec3f(1, 2, 3), GfVec3f(1, 3, 3),
        GfVec3f(4, 5, 6), GfVec3f(4, 6, 6)};
    reference.rest = reference.points;
    reference.curveId = {101, 202};
    reference.skinPrim = {7, 8};
    reference.skinPrimUv = {GfVec2f(.1f, .2f), GfVec2f(.3f, .4f)};
    desc.curveSets.push_back(std::move(reference));

    UsdGenNodeDesc source;
    source.path = desc.description.AppendChild(TfToken("Ops"))
        .AppendChild(TfToken("ReferenceSource"));
    source.type = TfToken("UsdGenReferenceSource");
    source.references = {SdfPath("/Reference/Guides")};

    UsdGenNodeDesc width;
    width.path = desc.description.AppendChild(TfToken("Ops"))
        .AppendChild(TfToken("Width"));
    width.type = TfToken("UsdGenWidth");
    width.inputs = {source.path};
    width.params.push_back({TfToken("width"), VtValue(.5f), false});

    desc.nodes = {source, width};
    desc.terminal = width.path;
    return desc;
}

static UsdGenGraphDesc ReferenceSourceLengthWidthDesc()
{
    auto desc = ReferenceSourceWidthDesc();
    UsdGenNodeDesc length;
    length.path = desc.description.AppendChild(TfToken("Ops"))
        .AppendChild(TfToken("Length"));
    length.type = TfToken("UsdGenLength");
    length.inputs = {desc.nodes[0].path};
    length.params = {
        {TfToken("length:mode"), VtValue(TfToken("scale")), false},
        {TfToken("length:value"), VtValue(.5f), false}};
    UsdGenNodeDesc terminal;
    terminal.path = desc.description.AppendChild(TfToken("Ops"))
        .AppendChild(TfToken("LengthWidth"));
    terminal.type = TfToken("UsdGenWidth");
    terminal.inputs = {length.path};
    terminal.params = {{TfToken("width"), VtValue(.75f), false}};
    UsdGenNodeDesc sibling;
    sibling.path = desc.description.AppendChild(TfToken("Ops"))
        .AppendChild(TfToken("LengthSibling"));
    sibling.type = TfToken("UsdGenWidth");
    sibling.inputs = {length.path};
    sibling.params = {{TfToken("width"), VtValue(.25f), false}};
    desc.nodes = {desc.nodes[0], length, terminal, sibling};
    desc.terminal = terminal.path;
    return desc;
}

struct AsyncCommitResult {
    std::atomic<bool> done{false};
    bool accepted = false;
    UsdGenSession::SnapshotPtr snapshot;
    UsdGenExecutionPipeline::Outcome outcome =
        UsdGenExecutionPipeline::Outcome::Superseded;
};

static std::shared_ptr<AsyncCommitResult>
SubmitRequest(UsdGenSession& session, UsdGenSession::CommitRequest request)
{
    auto result = std::make_shared<AsyncCommitResult>();
    result->accepted = session.CommitAsync(std::move(request), [result](
        UsdGenSession::SnapshotPtr snapshot,
        UsdGenExecutionPipeline::Outcome outcome) {
            result->snapshot = std::move(snapshot);
            result->outcome = outcome;
            result->done.store(true, std::memory_order_release);
        });
    return result;
}

static bool WaitFor(AsyncCommitResult const& result)
{
    auto const deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(10);
    while (!result.done.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    return result.done.load(std::memory_order_acquire);
}

static bool HasDiagnostic(UsdGenSession::SnapshotPtr const& snapshot,
                          char const* text)
{
    if (!snapshot) return false;
    for (std::string const& error : snapshot->diagnostics.errors)
        if (error.find(text) != std::string::npos) return true;
    return false;
}

struct SourceCallbackGateRelease {
    bool armed = false;
    ~SourceCallbackGateRelease() {
        if (armed) releaseCudaSourceAsyncCallbackGateForTesting();
    }
    void Release() {
        if (armed) releaseCudaSourceAsyncCallbackGateForTesting();
        armed = false;
    }
};

template <class T>
static bool ReadNamed(gpu::CudaNamedChannelLease const& lease,
                      cudaStream_t stream, std::vector<T>* values)
{
    if (!lease || !values || lease.Bytes().size % sizeof(T)) return false;
    values->resize(lease.Bytes().size / sizeof(T));
    return cudaMemcpyAsync(values->data(), lease.Bytes().data, lease.Bytes().size,
                           cudaMemcpyDeviceToHost, stream) == cudaSuccess &&
           cudaStreamSynchronize(stream) == cudaSuccess;
}

int main() {
    cudaStream_t consumer = nullptr;
    CHECK(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking) == cudaSuccess);
    // Authored named planes are uploaded into private buffers and published
    // with the same generation as their geometry. Retaining the first lease
    // proves that a later descriptor revision cannot mutate its COW data.
    {
        UsdGenSession session;
        auto desc = SourceDesc();
        UsdGenAuthoredPlaneDesc point;
        point.name = TfToken("density");
        point.type = UsdGenAuthoredPlaneType::Float32;
        point.domain = UsdGenAuthoredPlaneDomain::Point;
        point.arity = 1;
        point.floatValues = {1.f, 2.f, 3.f, 4.f, 5.f};
        UsdGenAuthoredPlaneDesc primitive;
        primitive.name = TfToken("classPair");
        primitive.type = UsdGenAuthoredPlaneType::Int32;
        primitive.domain = UsdGenAuthoredPlaneDomain::Primitive;
        primitive.arity = 2;
        primitive.intValues = {7, 8, 9, 10};
        UsdGenAuthoredPlaneDesc groom;
        groom.name = TfToken("groomWeight");
        groom.type = UsdGenAuthoredPlaneType::Float32;
        groom.domain = UsdGenAuthoredPlaneDomain::Groom;
        groom.floatValues = {0.25f};
        desc.curveSets.front().authoredPlanes = {point, primitive, groom};
        usdGenUsd::SetGraphDesc(session, desc);
        session.SetDevicePublicationEnabled(true);
        auto first = session.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(first && first->device && !session.LastDiagnostics().HasErrors() &&
              session.Stats().executionCacheMisses == 1 &&
              session.Stats().executionCacheAdmissions == 1);
        auto firstDensity = gpu::AcquireNamedChannel(first->device, TfToken("density"), consumer);
        auto firstClasses = gpu::AcquireNamedChannel(first->device, TfToken("classPair"), consumer);
        auto firstGroom = gpu::AcquireNamedChannel(first->device, TfToken("groomWeight"), consumer);
        std::vector<float> density;
        std::vector<int32_t> classes;
        std::vector<float> groomValue;
        CHECK(ReadNamed(firstDensity, consumer, &density) &&
              density == std::vector<float>({3.f, 4.f, 5.f, 1.f, 2.f}));
        CHECK(ReadNamed(firstClasses, consumer, &classes) &&
              classes == std::vector<int32_t>({9, 10, 7, 8}));
        CHECK(ReadNamed(firstGroom, consumer, &groomValue) &&
              groomValue == std::vector<float>({0.25f}));

        desc.curveSets.front().authoredPlanes.front().floatValues[0] = 99.f;
        usdGenUsd::SetGraphDesc(session, desc);
        auto second = session.Commit(2, UsdGenCommitReason::SetTime);
        CHECK(second && second->device && second != first);
        auto secondDensity = gpu::AcquireNamedChannel(second->device, TfToken("density"), consumer);
        CHECK(ReadNamed(secondDensity, consumer, &density) && density[3] == 99.f);
        CHECK(ReadNamed(firstDensity, consumer, &density) && density[3] == 1.f);
    }
    // Length compaction rebuilds every domain into private phase-5 outputs.
    // The short curve is culled after source canonicalization; Point and
    // Primitive payloads gather the surviving stable-id curve, while Groom
    // remains constant. A later generation cannot mutate the retained lease.
    {
        UsdGenSession session;
        auto desc = SourceDesc();
        UsdGenAuthoredPlaneDesc point;
        point.name = TfToken("density");
        point.type = UsdGenAuthoredPlaneType::Float32;
        point.domain = UsdGenAuthoredPlaneDomain::Point;
        point.arity = 1;
        point.floatValues = {1.f, 2.f, 3.f, 4.f, 5.f};
        UsdGenAuthoredPlaneDesc primitive;
        primitive.name = TfToken("classPair");
        primitive.type = UsdGenAuthoredPlaneType::Int32;
        primitive.domain = UsdGenAuthoredPlaneDomain::Primitive;
        primitive.arity = 2;
        primitive.intValues = {7, 8, 9, 10};
        UsdGenAuthoredPlaneDesc groom;
        groom.name = TfToken("groomWeight");
        groom.type = UsdGenAuthoredPlaneType::Float32;
        groom.domain = UsdGenAuthoredPlaneDomain::Groom;
        groom.arity = 1;
        groom.floatValues = {0.25f};
        desc.curveSets.front().authoredPlanes = {point, primitive, groom};
        UsdGenNodeDesc length;
        length.path = desc.description.AppendChild(TfToken("Ops"))
            .AppendChild(TfToken("Length"));
        length.type = TfToken("UsdGenLength");
        length.inputs = {desc.nodes.front().path};
        length.params = {
            {TfToken("length:mode"), VtValue(TfToken("cull")), false},
            {TfToken("cullThreshold"), VtValue(2.f), false}};
        desc.nodes.push_back(length);
        desc.terminal = length.path;
        usdGenUsd::SetGraphDesc(session, desc);
        session.SetDevicePublicationEnabled(true);
        auto compacted = session.Commit(10, UsdGenCommitReason::SetTime);
        CHECK(compacted && compacted->device &&
              !session.LastDiagnostics().HasErrors());
        auto geometry = gpu::AcquireGeometry(compacted->device, consumer);
        auto densityLease = gpu::AcquireNamedChannel(
            compacted->device, TfToken("density"), consumer);
        auto classLease = gpu::AcquireNamedChannel(
            compacted->device, TfToken("classPair"), consumer);
        auto groomLease = gpu::AcquireNamedChannel(
            compacted->device, TfToken("groomWeight"), consumer);
        std::vector<float> density;
        std::vector<int32_t> classes;
        std::vector<float> groomValue;
        CHECK(geometry && geometry.Geometry().curveCount == 1 &&
              geometry.Geometry().pointCount == 3);
        CHECK(densityLease.Metadata() && densityLease.Metadata()->elementCount == 3 &&
              ReadNamed(densityLease, consumer, &density) &&
              density == std::vector<float>({3.f, 4.f, 5.f}));
        CHECK(classLease.Metadata() && classLease.Metadata()->elementCount == 1 &&
              ReadNamed(classLease, consumer, &classes) &&
              classes == std::vector<int32_t>({9, 10}));
        CHECK(groomLease.Metadata() && groomLease.Metadata()->elementCount == 1 &&
              ReadNamed(groomLease, consumer, &groomValue) &&
              groomValue == std::vector<float>({0.25f}));

        desc.curveSets.front().authoredPlanes.front().floatValues[2] = 33.f;
        usdGenUsd::SetGraphDesc(session, desc);
        auto revised = session.Commit(11, UsdGenCommitReason::SetTime);
        CHECK(revised && revised->device && revised != compacted);
        auto revisedDensity = gpu::AcquireNamedChannel(
            revised->device, TfToken("density"), consumer);
        CHECK(ReadNamed(revisedDensity, consumer, &density) && density.front() == 33.f);
        CHECK(ReadNamed(densityLease, consumer, &density) && density.front() == 3.f);
    }
    // Source resampling rebuilds Point planes into private target-cardinality
    // buffers while Primitive data follows stable ids and Groom data stays
    // constant. A retained lease proves the next resampled publication cannot
    // mutate the first generation's COW payload.
    {
        UsdGenSession session;
        auto desc = SourceDesc();
        desc.nodes.front().params.push_back(
            {TfToken("resampleTo"), VtValue(4), false});
        UsdGenAuthoredPlaneDesc point;
        point.name = TfToken("density");
        point.type = UsdGenAuthoredPlaneType::Float32;
        point.domain = UsdGenAuthoredPlaneDomain::Point;
        point.arity = 1;
        point.floatValues = {10.f, 20.f, 30.f, 40.f, 70.f};
        UsdGenAuthoredPlaneDesc primitive;
        primitive.name = TfToken("classPair");
        primitive.type = UsdGenAuthoredPlaneType::Int32;
        primitive.domain = UsdGenAuthoredPlaneDomain::Primitive;
        primitive.arity = 2;
        primitive.intValues = {7, 8, 9, 10};
        UsdGenAuthoredPlaneDesc groom;
        groom.name = TfToken("groomWeight");
        groom.type = UsdGenAuthoredPlaneType::Float32;
        groom.domain = UsdGenAuthoredPlaneDomain::Groom;
        groom.floatValues = {0.25f};
        desc.curveSets.front().authoredPlanes = {point, primitive, groom};
        auto const originalDesc = desc;
        usdGenUsd::SetGraphDesc(session, desc);
        session.SetDevicePublicationEnabled(true);
        auto first = session.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(first && first->device && !session.LastDiagnostics().HasErrors());
        auto const firstOwner = first->device->Owner();
        auto firstDensity = gpu::AcquireNamedChannel(first->device, TfToken("density"), consumer);
        auto firstClasses = gpu::AcquireNamedChannel(first->device, TfToken("classPair"), consumer);
        auto firstGroom = gpu::AcquireNamedChannel(first->device, TfToken("groomWeight"), consumer);
        std::vector<float> density;
        std::vector<int32_t> classes;
        std::vector<float> groomValue;
        CHECK(ReadNamed(firstDensity, consumer, &density) && density.size() == 8);
        std::vector<float> const expected{30.f, 36.666667f, 50.f, 70.f,
                                          10.f, 13.333333f, 16.666667f, 20.f};
        CHECK(std::equal(density.begin(), density.end(), expected.begin(),
            [](float a, float b) { return std::fabs(a - b) < 1e-4f; }));
        CHECK(ReadNamed(firstClasses, consumer, &classes) &&
              classes == std::vector<int32_t>({9, 10, 7, 8}));
        CHECK(ReadNamed(firstGroom, consumer, &groomValue) &&
              groomValue == std::vector<float>({0.25f}));

        desc.curveSets.front().authoredPlanes.front().floatValues[0] = 99.f;
        usdGenUsd::SetGraphDesc(session, desc);
        auto second = session.Commit(2, UsdGenCommitReason::SetTime);
        CHECK(second && second->device && second != first);
        auto secondDensity = gpu::AcquireNamedChannel(second->device, TfToken("density"), consumer);
        CHECK(ReadNamed(secondDensity, consumer, &density) && density.size() == 8 &&
              std::fabs(density[4] - 99.f) < 1e-4f);
        CHECK(ReadNamed(firstDensity, consumer, &density) &&
              std::equal(density.begin(), density.end(), expected.begin(),
                  [](float a, float b) { return std::fabs(a - b) < 1e-4f; }));

        desc.curveSets.front().authoredPlanes.front().floatValues[0] = 55.f;
        failNextCudaSourceRelayNamedNativeCallbackForTesting();
        usdGenUsd::SetGraphDesc(session, desc);
        CHECK(session.Commit(3, UsdGenCommitReason::SetTime) == second &&
              session.LastDiagnostics().HasErrors());
        CHECK(ReadNamed(secondDensity, consumer, &density) && density.size() == 8 &&
              std::fabs(density[4] - 99.f) < 1e-4f);

        // Reverting to the exact first cache tuple after the native callback
        // failure must execute in a replacement workspace. The poisoned
        // context clears residency and advances compatibility identity, so it
        // cannot republish firstOwner even though that immutable COW owner and
        // its retained consumer lease remain valid.
        usdGenUsd::SetGraphDesc(session, originalDesc);
        auto recovered = session.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(recovered && recovered->device && recovered != first &&
              !session.LastDiagnostics().HasErrors() &&
              recovered->device->Owner() != firstOwner);
        CHECK(ReadNamed(firstDensity, consumer, &density) &&
              std::equal(density.begin(), density.end(), expected.begin(),
                  [](float a, float b) { return std::fabs(a - b) < 1e-4f; }));
    }
    // A backend-neutral context-loss fence cancels an in-flight CUDA
    // publication on the command owner, but cache/workspace mutation occurs
    // only later on the serialized cooker lane. Recovery of the exact cached
    // tuple must execute on a new owner while the old COW consumer stays valid.
    {
        int device = -1;
        CHECK(cudaGetDevice(&device) == cudaSuccess && device >= 0);
        UsdGenSession session(1);
        auto desc = WidthChainDesc();
        usdGenUsd::SetGraphDesc(session, desc);
        session.SetDevicePublicationEnabled(true);
        auto first = session.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(first && first->device && !session.LastDiagnostics().HasErrors());
        auto const oldOwner = first->device->Owner();
        auto oldLease = gpu::AcquireGeometry(first->device, consumer);
        CHECK(oldOwner && oldLease);
        auto const oldPoints = oldLease.Geometry().points.data;
        float3 oldPoint{};
        CHECK(cudaMemcpyAsync(&oldPoint, oldPoints, sizeof(oldPoint),
                              cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess);

        auto cached = session.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(cached && cached->device && cached != first &&
              cached->device->Owner() == oldOwner &&
              session.Stats().executionCacheHits == 1 &&
              session.Stats().executionCacheMisses == 1 &&
              session.Stats().executionCacheAdmissions == 1);

        // Loss signals are backend-scoped. Another backend's notification
        // cannot invalidate the active CUDA context, and the exact tuple still hits.
        CHECK(!session.PostDeviceContextLost(UsdGenBackendId::Invalid, device));
        CHECK(session.PostDeviceContextLost(usdGenTestUtils::kTestBackendB, device));
        auto backendFiltered = session.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(backendFiltered && backendFiltered->device &&
              backendFiltered->device->Owner() == oldOwner &&
              session.Stats().executionCacheHits == 2);

        armCudaSourceAsyncCallbackGateForTesting();
        SourceCallbackGateRelease releaseGate{true};
        UsdGenSession::CommitRequest request;
        request.frame = 2;
        request.reason = UsdGenCommitReason::SetTime;
        auto inFlight = SubmitRequest(session, std::move(request));
        CHECK(inFlight && inFlight->accepted);
        waitCudaSourceAsyncCallbackGateForTesting();
        session.NotifyDeviceContextLost(usdGenTestUtils::Backend("cuda"), device);
        CHECK(!inFlight->done.load(std::memory_order_acquire) &&
              session.NeedsCommit());
        releaseGate.Release();
        CHECK(WaitFor(*inFlight) &&
              inFlight->outcome == UsdGenExecutionPipeline::Outcome::Superseded &&
              session.Generation() == backendFiltered);

        auto recovered = session.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(recovered && recovered->device && recovered != backendFiltered &&
              !session.LastDiagnostics().HasErrors() &&
              recovered->device->Owner() != oldOwner &&
              session.Stats().executionCacheMisses == 2 &&
              session.Stats().executionCacheAdmissions == 2);
        auto recoveredLease = gpu::AcquireGeometry(recovered->device, consumer);
        CHECK(recoveredLease && recoveredLease.Geometry().points.data != oldPoints);

        float3 retainedPoint{};
        CHECK(cudaMemcpyAsync(&retainedPoint, oldPoints, sizeof(retainedPoint),
                              cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess &&
              retainedPoint.x == oldPoint.x && retainedPoint.y == oldPoint.y &&
              retainedPoint.z == oldPoint.z);
    }
    // ReferenceSource is a constrained CUDA source root: it consumes one
    // immutable reference curve set and hands its full geometry contract to a
    // normal COW Width stage.  Planning must preserve that logical source ->
    // Width dependency before a Session ever allocates a device workspace.
    {
        auto desc = ReferenceSourceWidthDesc();
        UsdGenDiagnostics diagnostics;
        auto plan = CompileCudaGraph(usdGenUsd::Lower(desc), &diagnostics);
        CHECK(plan && !diagnostics.HasErrors());
        auto metadata = GetCudaExecutionPlanMetadata(*plan);
        CHECK(metadata && metadata->Operators().size() == 2 &&
              metadata->Tasks().size() == 3 && metadata->TerminalTask() == 2);
        auto const& sourceTask = metadata->Tasks()[0];
        auto const& widthTask = metadata->Tasks()[1];
        CHECK(sourceTask.kind == UsdGenExecutionTaskKind::Source &&
              sourceTask.key == usdGenUsd::KeyOf(desc.nodes[0].path) &&
              sourceTask.type == TfToken("UsdGenReferenceSource") &&
              sourceTask.dependencies.empty() &&
              widthTask.kind == UsdGenExecutionTaskKind::Operator &&
              widthTask.key == usdGenUsd::KeyOf(desc.nodes[1].path) &&
              widthTask.type == TfToken("UsdGenWidth") &&
              widthTask.dependencies == std::vector<uint32_t>{0} &&
              metadata->Tasks()[2].dependencies == std::vector<uint32_t>({0, 1}));

        UsdGenSession session(1);
        usdGenUsd::SetGraphDesc(session, desc);
        session.SetDevicePublicationEnabled(true);
        auto first = session.Commit(3, UsdGenCommitReason::SetTime);
        CHECK(first && first->device && !session.LastDiagnostics().HasErrors());
        auto firstOwner = first->device->Owner();
        auto oldLease = gpu::AcquireGeometry(first->device, consumer);
        CHECK(firstOwner && oldLease);
        auto geometry = oldLease.Geometry();
        CHECK(geometry.curveCount == 2 && geometry.pointCount == 4 &&
              geometry.points.size == 4 && geometry.curveOffsets.size == 3 &&
              geometry.stableIds.size == 2 && oldLease.RootPrim().size == 2 &&
              oldLease.RootUV().size == 2);
        std::array<float3, 4> points{};
        std::array<float, 4> widths{};
        std::array<uint32_t, 3> offsets{};
        std::array<uint64_t, 2> ids{};
        std::array<int32_t, 2> rootPrim{};
        std::array<float2, 2> rootUv{};
        CHECK(cudaMemcpyAsync(points.data(), geometry.points.data,
                              sizeof(points), cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaMemcpyAsync(widths.data(), geometry.widths.data,
                              sizeof(widths), cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaMemcpyAsync(offsets.data(), geometry.curveOffsets.data,
                              sizeof(offsets), cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaMemcpyAsync(ids.data(), geometry.stableIds.data,
                              sizeof(ids), cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaMemcpyAsync(rootPrim.data(), oldLease.RootPrim().data,
                              sizeof(rootPrim), cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaMemcpyAsync(rootUv.data(), oldLease.RootUV().data,
                              sizeof(rootUv), cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess);
        CHECK(points[0].x == 1 && points[0].y == 2 && points[0].z == 3 &&
              points[3].x == 4 && points[3].y == 6 && points[3].z == 6 &&
              widths[0] == .5f && widths[1] == .5f &&
              widths[2] == .5f && widths[3] == .5f &&
              offsets[0] == 0 && offsets[1] == 2 && offsets[2] == 4 &&
              ids[0] == 101 && ids[1] == 202 &&
              rootPrim[0] == 7 && rootPrim[1] == 8 &&
              rootUv[0].x == .1f && rootUv[0].y == .2f &&
              rootUv[1].x == .3f && rootUv[1].y == .4f);

        auto cached = session.Commit(3, UsdGenCommitReason::SetTime);
        CHECK(cached && cached != first && cached->device &&
              cached->device != first->device &&
              cached->device->Owner() == firstOwner &&
              session.Stats().executionCacheHits == 1);

        // Reference curve generation is part of the exact cache input tuple.
        // Changing it forces a new source/owner, while the old consumer lease
        // remains readable and therefore cannot observe the replacement data.
        desc.curveSets.front().curveGeneration++;
        desc.curveSets.front().points.front() = GfVec3f(9, 8, 7);
        usdGenUsd::SetGraphDesc(session, desc);
        auto changed = session.Commit(3, UsdGenCommitReason::SetTime);
        CHECK(changed && changed->device && changed->device->Owner() != firstOwner &&
              session.Stats().executionCacheMisses == 2);
        float3 oldPoint{};
        CHECK(cudaMemcpyAsync(&oldPoint, geometry.points.data, sizeof(oldPoint),
                              cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess &&
              oldPoint.x == 1 && oldPoint.y == 2 && oldPoint.z == 3);

        // These checks compile only; they must reject before any CUDA workspace
        // or source upload allocation is attempted.
        UsdGenAuthoredPlaneDesc plane;
        plane.name = TfToken("unsupported");
        plane.type = UsdGenAuthoredPlaneType::Float32;
        plane.domain = UsdGenAuthoredPlaneDomain::Point;
        plane.floatValues = {1, 2, 3, 4};
        auto namedPlane = ReferenceSourceWidthDesc();
        namedPlane.curveSets.front().authoredPlanes.push_back(plane);
        diagnostics = {};
        CHECK(!CompileCudaGraph(usdGenUsd::Lower(namedPlane), &diagnostics) && diagnostics.HasErrors());
        auto multipleReferences = ReferenceSourceWidthDesc();
        auto anotherReference = multipleReferences.curveSets.front();
        anotherReference.path = SdfPath("/Reference/Other");
        multipleReferences.curveSets.push_back(std::move(anotherReference));
        multipleReferences.nodes.front().references.push_back(
            SdfPath("/Reference/Other"));
        diagnostics = {};
        CHECK(!CompileCudaGraph(usdGenUsd::Lower(multipleReferences), &diagnostics) && diagnostics.HasErrors());
        auto mapped = ReferenceSourceWidthDesc();
        mapped.nodes.front().maps = {SdfPath("/Maps/unsupported")};
        diagnostics = {};
        CHECK(!CompileCudaGraph(usdGenUsd::Lower(mapped), &diagnostics) && diagnostics.HasErrors());
        auto topologyChanging = ReferenceSourceWidthDesc();
        topologyChanging.nodes.front().params.push_back(
            {TfToken("resampleTo"), VtValue(3), false});
        diagnostics = {};
        CHECK(!CompileCudaGraph(usdGenUsd::Lower(topologyChanging), &diagnostics) && diagnostics.HasErrors());
    }
    // The topology-trunk ReferenceSource path also runs through the
    // asynchronous Session task graph. Hold the source callback at the
    // source->Length handoff to observe the literal-Length reservation's
    // Pending balance, then retain the old lease while a changed reference
    // generation is published.
    {
        auto desc = ReferenceSourceLengthWidthDesc();
        UsdGenSession session(1);
        usdGenUsd::SetGraphDesc(session, desc);
        session.SetDevicePublicationEnabled(true);
        int device = -1;
        CHECK(cudaGetDevice(&device) == cudaSuccess);
        auto pool = FindUsdGenExecutionResourcePool(
            {usdGenTestUtils::Backend("cuda"), device});
        CHECK(pool);
        auto const before = pool->Snapshot();

        armCudaSourceAsyncCallbackGateForTesting();
        SourceCallbackGateRelease releaseGate{true};
        UsdGenSession::CommitRequest firstRequest;
        firstRequest.frame = 12.0;
        firstRequest.reason = UsdGenCommitReason::SetTime;
        auto first = SubmitRequest(session, std::move(firstRequest));
        CHECK(first->accepted);
        waitCudaSourceAsyncCallbackGateForTesting();
        auto const during = pool->Snapshot();
        CHECK(during.byKind[static_cast<size_t>(
                  UsdGenExecutionResourceKind::Pending)] >
              before.byKind[static_cast<size_t>(
                  UsdGenExecutionResourceKind::Pending)] &&
              during.usedBytes > before.usedBytes);
        releaseGate.Release();
        CHECK(WaitFor(*first) &&
              first->outcome == UsdGenExecutionPipeline::Outcome::Published &&
              first->snapshot && first->snapshot->generation &&
              first->snapshot->generation->device);
        auto const firstGeneration = first->snapshot->generation;
        auto oldLease = gpu::AcquireGeometry(firstGeneration->device,
                                              consumer);
        CHECK(oldLease && oldLease.Geometry().pointCount == 4 &&
              oldLease.RootPrim().size == 2);
        float3 oldPoint{};
        CHECK(cudaMemcpyAsync(&oldPoint, oldLease.Geometry().points.data,
                              sizeof(oldPoint), cudaMemcpyDeviceToHost,
                              consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess);

        desc.curveSets.front().curveGeneration++;
        desc.curveSets.front().points.front() = GfVec3f(91, 82, 73);
        usdGenUsd::SetGraphDesc(session, desc);
        UsdGenSession::CommitRequest secondRequest;
        secondRequest.frame = 12.0;
        secondRequest.reason = UsdGenCommitReason::SetTime;
        auto second = SubmitRequest(session, std::move(secondRequest));
        CHECK(second->accepted && WaitFor(*second) &&
              second->outcome == UsdGenExecutionPipeline::Outcome::Published &&
              second->snapshot && second->snapshot->generation &&
              second->snapshot->generation->device &&
              second->snapshot->generation->device != firstGeneration->device);
        auto newLease = gpu::AcquireGeometry(
            second->snapshot->generation->device, consumer);
        CHECK(newLease);
        float3 revisedPoint{};
        CHECK(cudaMemcpyAsync(&revisedPoint, newLease.Geometry().points.data,
                              sizeof(revisedPoint), cudaMemcpyDeviceToHost,
                              consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess &&
              oldPoint.x == 1 && oldPoint.y == 2 && oldPoint.z == 3 &&
              revisedPoint.x == 91 && revisedPoint.y == 82 &&
              revisedPoint.z == 73);
        // Keep the consumer lease alive through the changed publication and
        // re-read it to prove the old COW owner was not overwritten.
        float3 retainedPoint{};
        CHECK(cudaMemcpyAsync(&retainedPoint,
                              oldLease.Geometry().points.data,
                              sizeof(retainedPoint), cudaMemcpyDeviceToHost,
                              consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess &&
              retainedPoint.x == 1 && retainedPoint.y == 2 &&
              retainedPoint.z == 3);
    }
    // Source resampling happens after the GPU source upload: literal targets
    // change topology without changing the authored host curve set.
    {
        UsdGenSession session;
        auto uniform = SourceDesc();
        uniform.nodes.front().params.push_back({TfToken("resampleTo"), VtValue(4), false});
        usdGenUsd::SetGraphDesc(session, uniform);
        session.SetDevicePublicationEnabled(true);
        auto four = session.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(four && four->device);
        auto lease = gpu::AcquireGeometry(four->device, consumer);
        CHECK(lease && lease.Geometry().curveCount == 2 && lease.Geometry().pointCount == 8);
        std::vector<float3> points(8);
        std::vector<uint32_t> offsets(3);
        CHECK(cudaMemcpyAsync(points.data(), lease.Geometry().points.data, points.size() * sizeof(float3), cudaMemcpyDeviceToHost, consumer) == cudaSuccess);
        CHECK(cudaMemcpyAsync(offsets.data(), lease.Geometry().curveOffsets.data, offsets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost, consumer) == cudaSuccess);
        CHECK(cudaStreamSynchronize(consumer) == cudaSuccess);
        CHECK(offsets == std::vector<uint32_t>({0,4,8}) &&
              std::fabs(points[0].x - 20.0f) < 1e-4f && std::fabs(points[1].x - 20.666666f) < 1e-4f &&
              std::fabs(points[2].x - 22.0f) < 1e-4f && std::fabs(points[3].x - 24.0f) < 1e-4f &&
              std::fabs(points[4].x - 10.0f) < 1e-4f && std::fabs(points[5].x - 10.333334f) < 1e-4f &&
              std::fabs(points[6].x - 10.666666f) < 1e-4f && std::fabs(points[7].x - 11.0f) < 1e-4f);
        auto const lastGood = four;
        auto invalid = uniform;
        invalid.nodes.front().params.front().value = VtValue(1);
        usdGenUsd::SetGraphDesc(session, invalid);
        CHECK(session.Commit(2, UsdGenCommitReason::SetTime) == lastGood && session.LastDiagnostics().HasErrors());
        invalid.nodes.front().params.front().value = VtValue(-2);
        usdGenUsd::SetGraphDesc(session, invalid);
        CHECK(session.Commit(3, UsdGenCommitReason::SetTime) == lastGood && session.LastDiagnostics().HasErrors());
        auto ragged = uniform;
        ragged.nodes.front().params.front().value = VtValue(0);
        usdGenUsd::SetGraphDesc(session, ragged);
        auto zero = session.Commit(4, UsdGenCommitReason::SetTime);
        CHECK(zero && zero != lastGood);
        auto zeroLease = gpu::AcquireGeometry(zero->device, consumer);
        CHECK(zeroLease && zeroLease.Geometry().pointCount == 5);
    }
    // Groom scalar Int32 controls are evaluated per commit, before source
    // resampling; an invalid evaluated value must retain the last publication.
    {
        UsdGenSession session;
        auto desc = SourceDesc();
        UsdGenExpressionDesc expression;
        expression.path = SdfPath("/Groom/Description/Expressions/resample");
        expression.source = "$frame > 1 ? 4 : 0";
        expression.outputs.push_back({TfToken("result"), TfToken("int"), ScalarShape(expr::ScalarType::Int32)});
        desc.expressions.push_back(expression);
        UsdGenExpressionBinding binding;
        binding.expression = expression.path;
        binding.destination = TfToken("resampleTo");
        binding.nativeType = TfToken("int");
        binding.destinationShape = ScalarShape(expr::ScalarType::Int32);
        binding.domain = expr::Domain::Groom;
        binding.literal = VtValue(0);
        // This literal would be invalid without the connected groom result;
        // frame 2 evaluates to four and must win before source validation.
        desc.nodes.front().params.push_back({TfToken("resampleTo"), VtValue(1), false});
        desc.nodes.front().expressionBindings.push_back(binding);
        usdGenUsd::SetGraphDesc(session, desc);
        session.SetDevicePublicationEnabled(true);
        auto ragged = session.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(ragged && ragged->device && ragged->device->Geometry().pointCount == 5);
        auto uniform = session.Commit(2, UsdGenCommitReason::SetTime);
        CHECK(uniform && uniform != ragged && uniform->device->Geometry().pointCount == 8);
        desc.expressions[0].source = "1";
        usdGenUsd::SetGraphDesc(session, desc);
        CHECK(session.Commit(3, UsdGenCommitReason::SetTime) == uniform && session.LastDiagnostics().HasErrors());
        desc.expressions[0].source = "-2";
        usdGenUsd::SetGraphDesc(session, desc);
        CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == uniform && session.LastDiagnostics().HasErrors());
    }
    {
        UsdGenSession session;
        auto desc = SourceDesc();
        desc.curveSets[0].points[0][2] = 5.0f; // independent authored rest remains z=0
        UsdGenExpressionDesc expression;
        expression.path = SdfPath("/Groom/Description/Expressions/useRest");
        expression.source = "$frame < 2";
        expression.outputs.push_back({TfToken("result"), TfToken("bool"), ScalarShape(expr::ScalarType::Bool)});
        desc.expressions.push_back(expression);
        UsdGenExpressionBinding binding;
        binding.expression = expression.path; binding.destination = TfToken("useRest");
        binding.nativeType = TfToken("bool"); binding.destinationShape = ScalarShape(expr::ScalarType::Bool);
        binding.domain = expr::Domain::Groom; binding.literal = VtValue(false);
        desc.nodes.front().params.push_back({TfToken("resampleTo"), VtValue(4), false});
        desc.nodes.front().expressionBindings.push_back(binding);
        {
            UsdGenSession literalSession; literalSession.SetDevicePublicationEnabled(true);
            usdGenUsd::SetGraphDesc(literalSession, desc);
            auto literalRest = literalSession.Commit(1, UsdGenCommitReason::SetTime);
            auto literalCurrent = literalSession.Commit(2, UsdGenCommitReason::SetTime);
            CHECK(literalRest && literalCurrent && literalRest->device && literalCurrent->device &&
                  literalRest != literalCurrent && literalRest->device->Geometry().pointCount == 8 &&
                  literalCurrent->device->Geometry().pointCount == 8 &&
                  !literalRest->device->Geometry().alreadyDeformed &&
                  literalCurrent->device->Geometry().alreadyDeformed);
        }
        UsdGenExpressionDesc resampleExpression;
        resampleExpression.path = SdfPath("/Groom/Description/Expressions/useRestResample");
        resampleExpression.source = "$frame < 2 ? 3 : 2";
        resampleExpression.outputs.push_back({TfToken("result"), TfToken("int"), ScalarShape(expr::ScalarType::Int32)});
        desc.expressions.push_back(resampleExpression);
        UsdGenExpressionBinding resampleBinding;
        resampleBinding.expression = resampleExpression.path; resampleBinding.destination = TfToken("resampleTo");
        resampleBinding.nativeType = TfToken("int"); resampleBinding.destinationShape = ScalarShape(expr::ScalarType::Int32);
        resampleBinding.domain = expr::Domain::Groom; resampleBinding.literal = VtValue(4);
        desc.nodes.front().expressionBindings.push_back(resampleBinding);
        usdGenUsd::SetGraphDesc(session, desc); session.SetDevicePublicationEnabled(true);
        auto rootFor91 = [](gpu::CudaGeometryLease const& lease, bool rest = false) {
            auto geometry = lease.Geometry();
            auto const* channel = rest ? geometry.restPoints.data : geometry.points.data;
            if (!lease || !geometry.stableIds.data || !geometry.curveOffsets.data || !channel)
                return float3{NAN, NAN, NAN};
            std::vector<uint64_t> ids(geometry.curveCount);
            std::vector<uint32_t> offsets(geometry.curveCount + 1);
            if (cudaMemcpy(ids.data(), geometry.stableIds.data, ids.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
                cudaMemcpy(offsets.data(), geometry.curveOffsets.data, offsets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess)
                return float3{NAN, NAN, NAN};
            for (size_t curve = 0; curve < ids.size(); ++curve) if (ids[curve] == 91) {
                if (offsets[curve] >= offsets[curve + 1] || offsets[curve] >= geometry.pointCount)
                    return float3{NAN, NAN, NAN};
                float3 point{};
                if (cudaMemcpy(&point, channel + offsets[curve], sizeof(point), cudaMemcpyDeviceToHost) != cudaSuccess)
                    return float3{NAN, NAN, NAN};
                return point;
            }
            return float3{NAN, NAN, NAN};
        };
        auto restGeneration = session.Commit(1, UsdGenCommitReason::SetTime);
        CHECK(restGeneration && restGeneration->device && !restGeneration->device->Geometry().alreadyDeformed &&
              restGeneration->device->Geometry().pointCount == 6);
        auto restLease = gpu::AcquireGeometry(restGeneration->device, nullptr);
        CHECK(restLease);
        float3 restPoint = rootFor91(restLease);
        CHECK(std::fabs(restPoint.z - 5.0f) < 1e-6f);
        CHECK(std::fabs(rootFor91(restLease, true).z) < 1e-6f);
        auto currentGeneration = session.Commit(2, UsdGenCommitReason::SetTime);
        CHECK(currentGeneration && currentGeneration != restGeneration && currentGeneration->device &&
              currentGeneration->device->Geometry().alreadyDeformed && currentGeneration->device->Geometry().pointCount == 4);
        auto currentLease = gpu::AcquireGeometry(currentGeneration->device, nullptr);
        CHECK(currentLease);
        float3 currentPoint = rootFor91(currentLease);
        CHECK(std::fabs(currentPoint.z - 5.0f) < 1e-6f);
        CHECK(std::fabs(rootFor91(currentLease, true).z) < 1e-6f);
        auto badType = desc;
        badType.nodes.front().expressionBindings[0].nativeType = TfToken("float");
        badType.nodes.front().expressionBindings[0].destinationShape = ScalarShape(expr::ScalarType::Float32);
        usdGenUsd::SetGraphDesc(session, badType);
        CHECK(session.Commit(3, UsdGenCommitReason::SetTime) == currentGeneration && session.LastDiagnostics().HasErrors());
        for (expr::Domain domain : {expr::Domain::Primitive, expr::Domain::Point}) {
            auto badDomain = desc;
            badDomain.nodes.front().expressionBindings[0].domain = domain;
            usdGenUsd::SetGraphDesc(session, badDomain);
            CHECK(session.Commit(3, UsdGenCommitReason::SetTime) == currentGeneration && session.LastDiagnostics().HasErrors());
        }
    }
    // The renderer selection belongs to the immutable request, not a
    // separately posted mutable side channel.  Start CUDA-enabled, reject two
    // false/absent requests against the retained device generation, then
    // recover and prove an absent request preserves the recovered true mode.
    {
        UsdGenSession session;
        auto desc = std::make_shared<const UsdGenGraphDesc>(SourceDesc());
        UsdGenSession::CommitRequest enable;
        enable.frame = 101.0;
        enable.reason = UsdGenCommitReason::SetTime;
        enable.spec = desc ? usdGenUsd::LowerShared(*desc) : nullptr;
        enable.devicePublication = true;
        auto first = SubmitRequest(session, std::move(enable));
        CHECK(first->accepted && WaitFor(*first));
        CHECK(first->outcome == UsdGenExecutionPipeline::Outcome::Published &&
              first->snapshot && first->snapshot->generation &&
              first->snapshot->generation->device &&
              first->snapshot->generation->frame == 101.0);
        auto const retained = first->snapshot->generation;

        UsdGenSession::CommitRequest disable;
        disable.frame = 102.0;
        disable.reason = UsdGenCommitReason::SetTime;
        disable.devicePublication = false;
        auto rejected = SubmitRequest(session, std::move(disable));
        CHECK(rejected->accepted && WaitFor(*rejected));
        CHECK(rejected->outcome == UsdGenExecutionPipeline::Outcome::Failed &&
              rejected->snapshot && rejected->snapshot->generation == retained &&
              HasDiagnostic(rejected->snapshot, "device-aware consumer"));

        UsdGenSession::CommitRequest absentFalse;
        absentFalse.frame = 103.0;
        absentFalse.reason = UsdGenCommitReason::SetTime;
        auto stillRejected = SubmitRequest(session, std::move(absentFalse));
        CHECK(stillRejected->accepted && WaitFor(*stillRejected));
        CHECK(stillRejected->outcome == UsdGenExecutionPipeline::Outcome::Failed &&
              stillRejected->snapshot &&
              stillRejected->snapshot->generation == retained &&
              HasDiagnostic(stillRejected->snapshot, "device-aware consumer"));

        UsdGenSession::CommitRequest recover;
        recover.frame = 104.0;
        recover.reason = UsdGenCommitReason::SetTime;
        recover.devicePublication = true;
        auto recovered = SubmitRequest(session, std::move(recover));
        CHECK(recovered->accepted && WaitFor(*recovered));
        CHECK(recovered->outcome == UsdGenExecutionPipeline::Outcome::Published &&
              recovered->snapshot && recovered->snapshot->generation &&
              recovered->snapshot->generation->device &&
              recovered->snapshot->generation != retained &&
              recovered->snapshot->generation->frame == 104.0);

        UsdGenSession::CommitRequest absentTrue;
        absentTrue.frame = 105.0;
        absentTrue.reason = UsdGenCommitReason::SetTime;
        auto stillEnabled = SubmitRequest(session, std::move(absentTrue));
        CHECK(stillEnabled->accepted && WaitFor(*stillEnabled));
        CHECK(stillEnabled->outcome == UsdGenExecutionPipeline::Outcome::Published &&
              stillEnabled->snapshot && stillEnabled->snapshot->generation &&
              stillEnabled->snapshot->generation->device &&
              stillEnabled->snapshot->generation != recovered->snapshot->generation &&
              stillEnabled->snapshot->generation->frame == 105.0);
    }
    gpu::CudaGeometryLease retained;
    std::weak_ptr<const UsdGenDeviceOwner> oldOwner;
    {
        UsdGenSession session;
        auto desc = SourceDesc();
        usdGenUsd::SetGraphDesc(session, desc);
        CHECK(!session.Commit(1, UsdGenCommitReason::SetTime));
        CHECK(session.LastDiagnostics().HasErrors() && session.NeedsCommit());
        session.SetDevicePublicationEnabled(true);
        auto first = session.Commit(1, UsdGenCommitReason::SetTime);
        if (!first) for (auto const& error : session.LastDiagnostics().errors)
            std::fprintf(stderr, "%s\n", error.c_str());
        CHECK(first && first->device && first->id == 0 && first->frame == 1);
        CHECK(!first->device->Geometry().alreadyDeformed);
        CHECK(usdGen::UsdGenCurvePublicationOf(*first).tiles.empty() && usdGen::UsdGenCurvePublicationOf(*first).guides.empty() && usdGen::UsdGenCurvePublicationOf(*first).instancers.empty());
        CHECK(!session.LastDiagnostics().HasErrors() && !session.LastDiagnostics().warnings.empty());
        CHECK(session.Graph().Spec().executionBackend == usdGenTestUtils::Backend("cuda"));
        retained = gpu::AcquireGeometry(first->device, consumer);
        CHECK(retained && retained.Geometry().pointCount == 5);
        oldOwner = first->device->Owner();
        auto copy = retained;
        retained = {};
        CHECK(copy && !oldOwner.expired());
        retained = std::move(copy);
        CHECK(!copy && copy.Geometry().points.data == nullptr);
        std::vector<float3> points(5);
        std::vector<float> widths(5);
        std::vector<uint32_t> offsets(3);
        std::vector<uint64_t> ids(2);
        auto view = retained.Geometry();
        CHECK(cudaMemcpyAsync(points.data(),view.points.data,5*sizeof(float3),cudaMemcpyDeviceToHost,consumer) == cudaSuccess);
        CHECK(cudaMemcpyAsync(widths.data(),view.widths.data,5*sizeof(float),cudaMemcpyDeviceToHost,consumer) == cudaSuccess);
        CHECK(cudaMemcpyAsync(offsets.data(),view.curveOffsets.data,3*sizeof(uint32_t),cudaMemcpyDeviceToHost,consumer) == cudaSuccess);
        CHECK(cudaMemcpyAsync(ids.data(),view.stableIds.data,2*sizeof(uint64_t),cudaMemcpyDeviceToHost,consumer) == cudaSuccess);
        CHECK(cudaStreamSynchronize(consumer) == cudaSuccess);
        CHECK(points[0].x == 20 && points[2].x == 24 && points[3].x == 10);
        CHECK(widths == std::vector<float>(5,.025f));
        CHECK(offsets == std::vector<uint32_t>({0,3,5}) && ids == std::vector<uint64_t>({37,91}));

        desc.curveSets[0].points[2] = GfVec3f(50,0,0);
        usdGenUsd::SetGraphDesc(session, desc);
        auto second = session.Commit(2, UsdGenCommitReason::SetTime);
        CHECK(second && second->device && second->id == 1 && second->device != first->device);
        first.reset();
        CHECK(!oldOwner.expired());
        // A new source allocation cannot mutate a retained prior generation.
        CHECK(cudaMemcpyAsync(points.data(),view.points.data,5*sizeof(float3),cudaMemcpyDeviceToHost,consumer) == cudaSuccess);
        CHECK(cudaStreamSynchronize(consumer) == cudaSuccess && points[0].x == 20);
        {
            auto next = gpu::AcquireGeometry(second->device, consumer);
            CHECK(next && next.Geometry().points.data != view.points.data);
            CHECK(cudaMemcpyAsync(points.data(),next.Geometry().points.data,5*sizeof(float3),cudaMemcpyDeviceToHost,consumer) == cudaSuccess);
            CHECK(cudaStreamSynchronize(consumer) == cudaSuccess && points[0].x == 50);
        }
        auto malformed = desc;
        malformed.curveSets[0].skinPrim[0] = 900;
        usdGenUsd::SetGraphDesc(session, malformed);
        auto repaired = session.Commit(3, UsdGenCommitReason::SetTime);
        CHECK(repaired && repaired != second && repaired->device &&
              !session.LastDiagnostics().HasErrors());
        {
            auto repairedLease = gpu::AcquireGeometry(repaired->device,consumer);
            CHECK(repairedLease && repairedLease.RootPrim().size == 2);
            int32_t repairedPrim[2]{};
            CHECK(cudaMemcpyAsync(repairedPrim,repairedLease.RootPrim().data,sizeof(repairedPrim),
                                  cudaMemcpyDeviceToHost,consumer) == cudaSuccess &&
                  cudaStreamSynchronize(consumer) == cudaSuccess);
            CHECK(repairedPrim[0] == 8 && repairedPrim[1] >= 0 && repairedPrim[1] < 9 &&
                  malformed.curveSets[0].skinPrim[0] == 900);
        }
        second = repaired; // Subsequent malformed inputs retain this last-good capture.
        malformed = desc;
        malformed.curveSets[0].restFromCurrentPoints = true;
        usdGenUsd::SetGraphDesc(session, malformed);
        CHECK(session.Commit(3, UsdGenCommitReason::SetTime) == second);
        CHECK(session.LastDiagnostics().HasErrors());
        malformed = desc;
        malformed.curveSets[0].points.pop_back();
        usdGenUsd::SetGraphDesc(session, malformed);
        CHECK(session.Commit(3, UsdGenCommitReason::SetTime) == second);
        CHECK(session.LastDiagnostics().HasErrors());
        CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == second);
        auto linear = desc;
        linear.curveSets[0].type = TfToken("linear");
        usdGenUsd::SetGraphDesc(session, linear);
        CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == second);
        CHECK(session.LastDiagnostics().HasErrors());
        auto varying = desc;
        varying.curveSets[0].widths = {.2f};
        varying.curveSets[0].widthsInterpolation = TfToken("varying");
        usdGenUsd::SetGraphDesc(session, varying);
        CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == second);
        CHECK(session.LastDiagnostics().HasErrors());
        auto unsupported = desc;
        unsupported.nodes[0].params.push_back({TfToken("useRest"), VtValue(std::string("false")), false});
        usdGenUsd::SetGraphDesc(session, unsupported);
        CHECK(session.Commit(5, UsdGenCommitReason::SetTime) == second);
        CHECK(session.LastDiagnostics().HasErrors());
        unsupported = desc;
        unsupported.nodes[0].type = TfToken("UsdGenNoise");
        usdGenUsd::SetGraphDesc(session, unsupported);
        CHECK(session.Commit(5, UsdGenCommitReason::SetTime) == second);
        CHECK(session.LastDiagnostics().HasErrors());
        usdGenUsd::SetGraphDesc(session, desc);
        session.SetDevicePublicationEnabled(false);
        CHECK(session.Commit(6, UsdGenCommitReason::SetTime) == second);
        CHECK(session.LastDiagnostics().HasErrors());
        auto cache = desc;
        cache.nodes[0].params.push_back({TfToken("useRest"), VtValue(false), false});
        cache.curveSets[0].rest.clear();
        usdGenUsd::SetGraphDesc(session, cache);
        session.SetDevicePublicationEnabled(true);
        auto cacheGeneration = session.Commit(7, UsdGenCommitReason::SetTime);
        CHECK(cacheGeneration && cacheGeneration->id == second->id + 1 &&
              cacheGeneration->device->Geometry().alreadyDeformed);
        auto empty = desc;
        auto& emptyCurves = empty.curveSets[0];
        emptyCurves.curveVertexCounts.clear(); emptyCurves.points.clear();
        emptyCurves.rest.clear(); emptyCurves.curveId.clear();
        emptyCurves.skinPrim.clear(); emptyCurves.skinPrimUv.clear();
        usdGenUsd::SetGraphDesc(session, empty);
        auto emptyGeneration = session.Commit(8, UsdGenCommitReason::SetTime);
        CHECK(emptyGeneration && emptyGeneration->id == cacheGeneration->id + 1 && emptyGeneration->device);
        auto emptyLease = gpu::AcquireGeometry(emptyGeneration->device, consumer);
        CHECK(emptyLease && emptyLease.Geometry().curveCount == 0 &&
              emptyLease.Geometry().pointCount == 0 && emptyLease.Geometry().curveOffsets.size == 1);
    }
    CHECK(retained && !oldOwner.expired());
    retained = {};
    CHECK(cudaStreamSynchronize(consumer) == cudaSuccess);
    CHECK(DrainCudaRetirement() && oldOwner.expired());

    // The async Session path retains decoded image data by immutable COW
    // ownership. A replacement descriptor can publish new texels while an
    // external lease keeps the prior device generation readable.
    {
        UsdGenSession session(1);
        auto desc = ImageWidthDesc();
        usdGenUsd::SetGraphDesc(session, desc);
        session.SetDevicePublicationEnabled(true);
        UsdGenSession::CommitRequest request;
        request.frame = 1;
        request.reason = UsdGenCommitReason::SetTime;
        auto first = SubmitRequest(session, std::move(request));
        CHECK(first->accepted && WaitFor(*first) &&
              first->outcome == UsdGenExecutionPipeline::Outcome::Published &&
              first->snapshot && first->snapshot->generation &&
              first->snapshot->generation->device);
        auto oldLease = gpu::AcquireGeometry(
            first->snapshot->generation->device, consumer);
        CHECK(oldLease && oldLease.Geometry().widths.size == 5);
        std::vector<float> oldWidths(5), newWidths(5);
        CHECK(cudaMemcpyAsync(oldWidths.data(), oldLease.Geometry().widths.data,
                              oldWidths.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess);
        // No CUDA operator samples a map today, so Width's envelope is the
        // authored usdGen:mask (1) and every width is the plain target.  The
        // map still participates in the plan's identity, which is what the
        // COW-payload replacement below exercises.
        for (float value : oldWidths)
            CHECK(std::fabs(value - .8f) < 2e-5f);

        auto const oldPayload = desc.maps.front().imagePayload;
        desc.maps.front().textureGeneration++;
        desc.maps.front().imagePayload = UsdGenImagePayload::Create(
            1, 1, 1, std::vector<float>{.25f},
            UsdGenImageRowOrientation::BottomUp);
        usdGenUsd::SetGraphDesc(session, desc);
        request = {};
        request.frame = 1;
        request.reason = UsdGenCommitReason::SetTime;
        auto replacement = SubmitRequest(session, std::move(request));
        CHECK(replacement->accepted && WaitFor(*replacement) &&
              replacement->outcome == UsdGenExecutionPipeline::Outcome::Published &&
              replacement->snapshot && replacement->snapshot->generation &&
              replacement->snapshot->generation->device &&
              oldPayload && oldPayload->Data()[0] == .5f);
        auto newLease = gpu::AcquireGeometry(
            replacement->snapshot->generation->device, consumer);
        CHECK(newLease && newLease.Geometry().widths.size == 5);
        CHECK(cudaMemcpyAsync(newWidths.data(), newLease.Geometry().widths.data,
                              newWidths.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess);
        for (float value : newWidths)
            CHECK(std::fabs(value - .8f) < 2e-5f);
        CHECK(cudaMemcpyAsync(oldWidths.data(), oldLease.Geometry().widths.data,
                              oldWidths.size() * sizeof(float),
                              cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess);
        for (float value : oldWidths)
            CHECK(std::fabs(value - .8f) < 2e-5f);
    }

    // A compiled Source -> Width -> Width chain is reusable across frames.
    // Keep the first width generation leased while later commits replace the
    // session's published generation, proving that its borrowed device data
    // remains valid until the consumer releases it.
    gpu::CudaGeometryLease retainedWidths;
    std::weak_ptr<const UsdGenDeviceOwner> widthOwner;
    {
        UsdGenSession session;
        auto desc = WidthChainDesc();
        usdGenUsd::SetGraphDesc(session, desc);
        session.SetDevicePublicationEnabled(true);
        auto first = session.Commit(1, UsdGenCommitReason::SetTime);
        if (!first) for (auto const& error : session.LastDiagnostics().errors)
            std::fprintf(stderr, "%s\n", error.c_str());
        CHECK(first && first->device && first->id == 0 && first->frame == 1);
        auto const* plan = session.Graph().DevicePlan().get();
        CHECK(plan != nullptr);
        retainedWidths = gpu::AcquireGeometry(first->device, consumer);
        CHECK(retainedWidths && retainedWidths.Geometry().pointCount == 5);
        widthOwner = first->device->Owner();
        auto firstView = retainedWidths.Geometry();
        std::vector<float> widths(5);
        auto readWidths = [&](gpu::DeviceCurveGeometryView view) {
            if (cudaMemcpyAsync(widths.data(), view.widths.data,
                                widths.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, consumer) != cudaSuccess)
                return false;
            return cudaStreamSynchronize(consumer) == cudaSuccess;
        };

        // WidthLiteral supplies a literal width of .1. WidthExpr then uses a
        // typed point field, primitive replace, and groom enabled field.
        CHECK(readWidths(firstView));
        float const pointFrame1 = 0.1f + 1.0f / 24.0f;
        CHECK(std::fabs(widths[0] - pointFrame1) < 2e-5f);
        CHECK(std::fabs(widths[1] - pointFrame1) < 2e-5f);
        CHECK(std::fabs(widths[2] - pointFrame1) < 2e-5f);
        CHECK(std::fabs(widths[3] - 0.1f * pointFrame1) < 2e-5f);
        CHECK(std::fabs(widths[4] - 0.1f * pointFrame1) < 2e-5f);

        auto second = session.Commit(2, UsdGenCommitReason::SetTime);
        CHECK(second && second->device && second->id == 1 && second->frame == 2);
        CHECK(session.Graph().DevicePlan().get() == plan);
        // Primitive 0 has replace=true; primitive 1 has replace=false. The
        // latter multiplies WidthLiteral's .1 by the point target.
        auto secondLease = gpu::AcquireGeometry(second->device, consumer);
        CHECK(secondLease);
        CHECK(readWidths(secondLease.Geometry()));
        float const pointFrame2 = 0.2f + 2.0f / 24.0f;
        CHECK(std::fabs(widths[0] - pointFrame2) < 2e-5f);
        CHECK(std::fabs(widths[1] - pointFrame2) < 2e-5f);
        CHECK(std::fabs(widths[2] - pointFrame2) < 2e-5f);
        CHECK(std::fabs(widths[3] - 0.1f * pointFrame2) < 2e-5f);
        CHECK(std::fabs(widths[4] - 0.1f * pointFrame2) < 2e-5f);
        CHECK(!widthOwner.expired());
        first.reset();
        CHECK(readWidths(firstView));
        CHECK(std::fabs(widths[0] - pointFrame1) < 2e-5f);
        CHECK(std::fabs(widths[3] - 0.1f * pointFrame1) < 2e-5f);

        auto third = session.Commit(3, UsdGenCommitReason::SetTime);
        CHECK(third && third->device && third->id == 2 && third->frame == 3);
        // enabled is false at frame 3, so WidthExpr is an exact passthrough
        // and the preceding literal Width remains .1 everywhere.
        auto thirdLease = gpu::AcquireGeometry(third->device, consumer);
        CHECK(thirdLease);
        CHECK(readWidths(thirdLease.Geometry()));
        for (float width : widths) CHECK(width == 0.1f);

        // A malformed program must leave the last-good generation and plan
        // published. A runtime non-finite result must do the same.
        auto lastGood = third;
        auto malformed = desc;
        malformed.expressions[0].source = "$frame +";
        usdGenUsd::SetGraphDesc(session, malformed);
        CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == lastGood);
        CHECK(session.LastDiagnostics().HasErrors() && session.NeedsCommit());
        CHECK(session.Graph().DevicePlan().get() == plan);

        auto runtimeBad = desc;
        runtimeBad.expressions[0].source = "$frame / ($frame - 4)";
        usdGenUsd::SetGraphDesc(session, runtimeBad);
        CHECK(session.Commit(4, UsdGenCommitReason::SetTime) == lastGood);
        CHECK(session.LastDiagnostics().HasErrors() && session.NeedsCommit());
        // Compilation succeeded, so Graph()/DevicePlan() may now describe the
        // attempted program even though publication correctly retained the
        // last-good generation.

        usdGenUsd::SetGraphDesc(session, desc);
        auto recovered = session.Commit(5, UsdGenCommitReason::SetTime);
        CHECK(recovered && recovered->device && recovered != lastGood && recovered->frame == 5);

        // Empty authored tiles still produce a valid empty Source -> Width
        // generation. The point binding is intentionally retained: empty
        // domains must not force a null field dereference or a stale publish.
        auto empty = desc;
        auto& curves = empty.curveSets[0];
        curves.curveVertexCounts.clear();
        curves.points.clear();
        curves.rest.clear();
        curves.curveId.clear();
        curves.skinPrim.clear();
        curves.skinPrimUv.clear();
        usdGenUsd::SetGraphDesc(session, empty);
        auto emptyGeneration = session.Commit(6, UsdGenCommitReason::SetTime);
        for (auto const& error : session.LastDiagnostics().errors)
            std::fprintf(stderr, "%s\n", error.c_str());
        CHECK(emptyGeneration && emptyGeneration->device && emptyGeneration != recovered);
        CHECK(!session.LastDiagnostics().HasErrors());
        auto emptyLease = gpu::AcquireGeometry(emptyGeneration->device, consumer);
        CHECK(emptyLease && emptyLease.Geometry().curveCount == 0 &&
              emptyLease.Geometry().pointCount == 0 &&
              emptyLease.Geometry().curveOffsets.size == 1 &&
              emptyLease.Geometry().widths.size == 0);
    }
    CHECK(retainedWidths && !widthOwner.expired());
    retainedWidths = {};
    CHECK(cudaStreamSynchronize(consumer) == cudaSuccess);
    CHECK(DrainCudaRetirement() && widthOwner.expired());

    // A cooker driven directly through the CUDA provider uses the same
    // device-aware cache contract as the session owner: a successful device generation first becomes a deferred
    // candidate, and only an accepted candidate enters residency.  A hit then
    // republishes a fresh host/device identity while retaining the immutable
    // COW owner and its native allocations.
    std::weak_ptr<const UsdGenDeviceOwner> cachedCookerOwner;
    {
        int device = -1;
        CHECK(cudaGetDevice(&device) == cudaSuccess && device >= 0);
        auto cacheDomain = std::make_shared<UsdGenExecutionCacheDomain>(
            UsdGenExecutionCacheDomainKey{usdGenTestUtils::Backend("cuda"), device, 0},
            64u * 1024u * 1024u);
        auto cookerProvider = CudaProvider();
        CHECK(cookerProvider != nullptr);
        UsdGenSessionCooker cooker(2, 64u * 1024u * 1024u, cacheDomain);
        auto cudaDesc = std::make_shared<const UsdGenGraphDesc>(WidthChainDesc());
        auto first = CookCuda(cooker, cookerProvider, usdGenUsd::LowerShared(*cudaDesc), 1.0, {}, {}, device);
        auto firstCandidate = cooker.TakeCacheCandidate();
        CHECK(first && first->device && first->id == 0 &&
              first->device->Identity().generation == 0 && firstCandidate &&
              cooker.ExecutionCacheSize() == 0 &&
              cooker.Stats().executionCacheMisses == 1);
        auto firstOwner = first->device->Owner();
        cachedCookerOwner = firstOwner;
        size_t const firstBytes = first->device->ExclusiveRetainedBytes();
        CHECK(firstOwner && firstOwner->ProducerReady() && firstBytes > 0 &&
              firstCandidate->bytes > firstBytes);
        auto mischargedCandidate = *firstCandidate;
        ++mischargedCandidate.bytes;
        CHECK(!cooker.CommitCacheCandidate(mischargedCandidate) &&
              cooker.ExecutionCacheSize() == 0 &&
              cooker.ExecutionCacheBytes() == 0 &&
              cooker.Stats().executionCacheAdmissionFailures == 1);
        CHECK(cooker.CommitCacheCandidate(*firstCandidate) &&
              cooker.ExecutionCacheSize() == 1 &&
              cooker.ExecutionCacheBytes() == firstCandidate->bytes &&
              cooker.Stats().executionCacheAdmissions == 1);
        size_t const residentBytes = cooker.ExecutionCacheBytes();
        UsdGenStats const afterAdmission = cooker.Stats();

        // A point revision is a real COW overlay: its owner retains the base
        // generation while owning only the replacement points itself. The
        // exclusive value remains delta-only, but the cache-charge value must
        // include the reachable base even if the base cache record later
        // disappears.
        auto replacementPoints = std::make_unique<gpu::DeviceBuffer<float3>>();
        CHECK(replacementPoints->reset(first->device->Geometry().pointCount) ==
              cudaSuccess);
        CHECK(cudaMemset(replacementPoints->data(), 0,
                         replacementPoints->bytes()) == cudaSuccess &&
              replacementPoints->recordUse(nullptr) == cudaSuccess);
        std::string revisionReason;
        auto revision = gpu::MakePointRevisionGeneration(
            first->device, 99, std::move(replacementPoints), {},
            &revisionReason);
        CHECK(revision && revision->Owner() && revision->Owner() != firstOwner &&
              revision->ExclusiveRetainedBytes() > 0 &&
              revision->InclusiveRetainedBytes() >=
                  first->device->InclusiveRetainedBytes() +
                  revision->ExclusiveRetainedBytes());

        auto second = CookCuda(cooker, cookerProvider, usdGenUsd::LowerShared(*cudaDesc), 1.0, first, afterAdmission, device);
        CHECK(second && second != first && second->device && second->id == 1 &&
              second->device->Identity().generation == 1 &&
              second->device->Geometry().valueVersion == 1 &&
              second->device->Owner() == firstOwner &&
              second->device->ExclusiveRetainedBytes() == firstBytes &&
              !cooker.TakeCacheCandidate() && cooker.ExecutionCacheSize() == 1 &&
              cooker.ExecutionCacheBytes() == residentBytes &&
              cooker.Stats().executionCacheHits == 1 &&
              cooker.Stats().executionCacheMisses == 1);
        CHECK(!cooker.CommitCacheCandidate(*firstCandidate));

        auto frameMiss = CookCuda(cooker, cookerProvider, usdGenUsd::LowerShared(*cudaDesc), 2.0, second, {}, device);
        auto frameCandidate = cooker.TakeCacheCandidate();
        CHECK(frameMiss && frameMiss != second && frameMiss->device &&
              frameMiss->id == 2 && frameMiss->frame == 2.0 && frameCandidate &&
              cooker.ExecutionCacheSize() == 1 &&
              cooker.ExecutionCacheBytes() == residentBytes);

        auto changed = WidthChainDesc();
        changed.nodes[1].params.front().value = VtValue(0.2f);
        auto changedDesc = std::make_shared<const UsdGenGraphDesc>(
            std::move(changed));
        auto parameterMiss = CookCuda(cooker, cookerProvider, usdGenUsd::LowerShared(*changedDesc), 2.0, frameMiss, {}, device);
        auto parameterCandidate = cooker.TakeCacheCandidate();
        CHECK(parameterMiss && parameterMiss != frameMiss &&
              parameterMiss->device && parameterMiss->id == 3 &&
              parameterCandidate && cooker.ExecutionCacheSize() == 1 &&
              cooker.ExecutionCacheBytes() == residentBytes);
        CHECK(frameCandidate &&
              !cooker.CommitCacheCandidate(*frameCandidate));
        CHECK(parameterCandidate &&
              cooker.CommitCacheCandidate(*parameterCandidate) &&
              cooker.ExecutionCacheSize() == 2);
        revision.reset();
    }

    // A completed CUDA result may cross cooker/Session ownership only through
    // the shared domain's immutable COW snapshot.  The receiving cooker gets
    // a new public generation and device wrapper, while the native owner is
    // deliberately shared.  A domain-epoch replacement then makes both a
    // queued old candidate and the former resident tuple unusable without
    // invalidating a consumer which already acquired its own geometry lease.
    {
        int device = -1;
        CHECK(cudaGetDevice(&device) == cudaSuccess && device >= 0);
        auto sharedDomain = std::make_shared<UsdGenExecutionCacheDomain>(
            UsdGenExecutionCacheDomainKey{
                usdGenTestUtils::Backend("cuda"), device, 0x434f575f43414348ull},
            64u * 1024u * 1024u);
        auto cudaDesc = std::make_shared<const UsdGenGraphDesc>(WidthChainDesc());
        auto leaderProvider = CudaProvider();
        auto followerProvider = CudaProvider();
        CHECK(leaderProvider && followerProvider);
        UsdGenSessionCooker leader(2, 64u * 1024u * 1024u, sharedDomain);
        UsdGenSessionCooker follower(2, 64u * 1024u * 1024u, sharedDomain);

        auto first = CookCuda(leader, leaderProvider, usdGenUsd::LowerShared(*cudaDesc), 7.0, {}, {}, device);
        auto firstCandidate = leader.TakeCacheCandidate();
        CHECK(first && first->device && firstCandidate &&
              leader.CommitCacheCandidate(*firstCandidate) &&
              sharedDomain->Size() == 1);
        auto const firstOwner = first->device->Owner();

        auto reused = CookCuda(follower, followerProvider, usdGenUsd::LowerShared(*cudaDesc), 7.0, {}, {}, device);
        CHECK(reused && reused != first && reused->device &&
              reused->device != first->device &&
              reused->device->Owner() == firstOwner &&
              !follower.TakeCacheCandidate() &&
              follower.Stats().executionCacheHits == 1 &&
              follower.Stats().executionCacheMisses == 0 &&
              sharedDomain->Size() == 1);
        auto oldLease = gpu::AcquireGeometry(reused->device, consumer);
        CHECK(oldLease);
        auto const oldPoints = oldLease.Geometry().points.data;
        float3 oldPoint{};
        CHECK(cudaMemcpyAsync(&oldPoint, oldPoints, sizeof(oldPoint),
                              cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess);

        // Leave a real, unaccepted candidate in the old epoch.  Invalidation
        // must clear residency first and the stale candidate must not seed it
        // again, even though its producer generation remains current locally.
        auto pending = CookCuda(leader, leaderProvider, usdGenUsd::LowerShared(*cudaDesc), 8.0, first, leader.Stats(), device);
        auto staleCandidate = leader.TakeCacheCandidate();
        CHECK(pending && pending->device && staleCandidate &&
              sharedDomain->Invalidate() && sharedDomain->Size() == 0 &&
              !leader.CommitCacheCandidate(*staleCandidate) &&
              sharedDomain->Size() == 0);

        float3 retainedPoint{};
        CHECK(cudaMemcpyAsync(&retainedPoint, oldPoints, sizeof(retainedPoint),
                              cudaMemcpyDeviceToHost, consumer) == cudaSuccess &&
              cudaStreamSynchronize(consumer) == cudaSuccess &&
              retainedPoint.x == oldPoint.x && retainedPoint.y == oldPoint.y &&
              retainedPoint.z == oldPoint.z);

        auto recovered = CookCuda(follower, followerProvider, usdGenUsd::LowerShared(*cudaDesc), 7.0, reused, follower.Stats(), device);
        auto recoveredLease = recovered && recovered->device
            ? gpu::AcquireGeometry(recovered->device, consumer)
            : gpu::CudaGeometryLease{};
        CHECK(recovered && recovered->device && recovered != reused &&
              recovered->device != reused->device &&
              recovered->device->Owner() != firstOwner &&
              recoveredLease &&
              recoveredLease.Geometry().points.data != oldPoints &&
              follower.Stats().executionCacheMisses == 1);
    }

    // Two CUDA Sessions sharing one exact domain join while the leader's
    // source callback is held. The follower must not launch a second graph;
    // its wrapper is distinct while the immutable device owner is shared.
    {
        int device = -1;
        CHECK(cudaGetDevice(&device) == cudaSuccess && device >= 0);
        auto sharedDomain = std::make_shared<UsdGenExecutionCacheDomain>(
            UsdGenExecutionCacheDomainKey{
                usdGenTestUtils::Backend("cuda"), device, 0},
            64u * 1024u * 1024u);
        auto sharedDesc = usdGenUsd::LowerShared(WidthChainDesc());
        UsdGenSession leader(2, 4096, sharedDomain);
        UsdGenSession follower(2, 4096, sharedDomain);
        UsdGenSession::CommitRequest leaderRequest;
        leaderRequest.spec = sharedDesc;
        leaderRequest.devicePublication = true;
        leaderRequest.callerDevice = device;
        leaderRequest.frame = 17.0;
        leaderRequest.reason = UsdGenCommitReason::SetTime;
        UsdGenSession::CommitRequest followerRequest = leaderRequest;
        armCudaSourceAsyncCallbackGateForTesting();
        SourceCallbackGateRelease releaseGate{true};
        auto leaderResult = SubmitRequest(leader, std::move(leaderRequest));
        CHECK(leaderResult && leaderResult->accepted);
        waitCudaSourceAsyncCallbackGateForTesting();
        auto followerResult = SubmitRequest(follower, std::move(followerRequest));
        CHECK(followerResult && followerResult->accepted);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        CHECK(!followerResult->done.load(std::memory_order_acquire));
        releaseGate.Release();
        CHECK(WaitFor(*leaderResult) && WaitFor(*followerResult) &&
              leaderResult->outcome == UsdGenExecutionPipeline::Outcome::Published &&
              followerResult->outcome == UsdGenExecutionPipeline::Outcome::Published);
        CHECK(leaderResult->snapshot && followerResult->snapshot &&
              leaderResult->snapshot->generation && followerResult->snapshot->generation &&
              leaderResult->snapshot->generation != followerResult->snapshot->generation &&
              leaderResult->snapshot->generation->device !=
                  followerResult->snapshot->generation->device &&
              leaderResult->snapshot->generation->device->Owner() ==
                  followerResult->snapshot->generation->device->Owner() &&
              followerResult->snapshot->stats.executionCacheCoalesced == 1);
    }
    {
        int device = -1;
        CHECK(cudaGetDevice(&device) == cudaSuccess && device >= 0);
        auto shutdownDomain = std::make_shared<UsdGenExecutionCacheDomain>(
            UsdGenExecutionCacheDomainKey{
                usdGenTestUtils::Backend("cuda"), device, 0},
            64u * 1024u * 1024u);
        auto shutdownDesc = usdGenUsd::LowerShared(WidthChainDesc());
        UsdGenSession leader(2, 4096, shutdownDomain);
        std::atomic<bool> followerCallback{false};
        armCudaSourceAsyncCallbackGateForTesting();
        SourceCallbackGateRelease releaseGate{true};
        UsdGenSession::CommitRequest leaderRequest;
        leaderRequest.spec = shutdownDesc;
        leaderRequest.devicePublication = true;
        leaderRequest.callerDevice = device;
        leaderRequest.frame = 18.0;
        auto leaderResult = SubmitRequest(leader, std::move(leaderRequest));
        CHECK(leaderResult && leaderResult->accepted);
        waitCudaSourceAsyncCallbackGateForTesting();
        {
            UsdGenSession follower(2, 4096, shutdownDomain);
            UsdGenSession::CommitRequest request;
            request.spec = shutdownDesc;
            request.devicePublication = true;
            request.callerDevice = device;
            request.frame = 18.0;
            CHECK(follower.CommitAsync(std::move(request),
                [&](UsdGenSession::SnapshotPtr,
                    UsdGenExecutionPipeline::Outcome) {
                    followerCallback.store(true, std::memory_order_release);
                }));
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(followerCallback.load(std::memory_order_acquire));
        releaseGate.Release();
        CHECK(WaitFor(*leaderResult));
    }
    CHECK(cudaStreamSynchronize(consumer) == cudaSuccess);
    CHECK(DrainCudaRetirement() && cachedCookerOwner.expired());
    CHECK(cudaStreamDestroy(consumer) == cudaSuccess);
    std::puts("testUsdGenCudaSession: PASS");
    return 0;
}
