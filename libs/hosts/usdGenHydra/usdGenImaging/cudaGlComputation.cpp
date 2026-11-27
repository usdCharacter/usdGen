#include "usdGenImaging/cudaGlComputation.h"

#ifdef USDGEN_HAS_CUDA_GL_INTEROP

#include "usdGen/api/pipeline.h"
#include "usdGen/api/ext/backendDef.h"

#include "pxr/imaging/hd/bufferSpec.h"
#include "pxr/imaging/hd/types.h"
#include "pxr/imaging/hdSt/bufferArrayRange.h"
#include "pxr/imaging/hdSt/bufferResource.h"
#include "pxr/imaging/hgiGL/buffer.h"
#include "pxr/base/tf/diagnostic.h"
#include "pxr/imaging/garch/glApi.h"

#include <cuda_gl_interop.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <exception>
#include <sstream>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {
namespace {

bool
_Mul(size_t a, size_t b, size_t *result)
{
    if (a && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    *result = a * b;
    return true;
}

bool
_Add(size_t a, size_t b, size_t *result)
{
    if (b > std::numeric_limits<size_t>::max() - a) {
        return false;
    }
    *result = a + b;
    return true;
}

bool
_FitsInt(size_t value)
{
    return value <= static_cast<size_t>(std::numeric_limits<int>::max());
}

bool
_ToSize(uint64_t value, size_t *result)
{
    if (value > std::numeric_limits<size_t>::max()) {
        return false;
    }
    *result = static_cast<size_t>(value);
    return true;
}

} // namespace

UsdGenCudaGlComputation::UsdGenCudaGlComputation(
    std::shared_ptr<const usdGen::UsdGenDeviceGeneration> generation,
    usdGen::UsdGenDeviceChannelSemantic semantic,
    TfToken destinationName,
    std::optional<uint32_t> tileId)
    : _generation(std::move(generation))
    , _semantic(semantic)
    , _destinationName(std::move(destinationName))
    , _tileId(tileId)
{
}

void
UsdGenCudaGlComputation::_Fail(char const *where, int cudaError) noexcept
{
    std::ostringstream stream;
    stream << where << ": " << cudaGetErrorName(static_cast<cudaError_t>(cudaError))
           << " (" << cudaGetErrorString(static_cast<cudaError_t>(cudaError)) << ')';
    _Fail(stream.str());
}

void
UsdGenCudaGlComputation::_Fail(std::string message) noexcept
{
    _succeeded = false;
    _error = std::move(message);
}

bool
UsdGenCudaGlComputation::_WholeChannelCount(uint64_t *count) const noexcept
{
    if (!_generation || !count) {
        return false;
    }
    using S = usdGen::UsdGenDeviceChannelSemantic;
    switch (_semantic) {
    case S::Points:
    case S::RestPoints:
    case S::Widths:
    case S::HairT:
        *count = _generation->Geometry().pointCount;
        return true;
    case S::ElementOffsets:
        if (_generation->Geometry().elementCount ==
            std::numeric_limits<uint64_t>::max()) {
            return false;
        }
        *count = _generation->Geometry().elementCount + 1;
        return true;
    case S::StableIds:
    case S::RootPrim:
    case S::RootUV:
        *count = _generation->Geometry().elementCount;
        return true;
    default:
        return false;
    }
}

bool
UsdGenCudaGlComputation::_SelectedTile(
    usdGen::UsdGenDeviceTileMetadata *tile) const noexcept
{
    if (!_tileId || !_generation || !tile) {
        return false;
    }

    auto const &geometry = _generation->Geometry();
    uint64_t curveEnd = 0;
    uint64_t pointEnd = 0;
    uint32_t previousId = 0;
    bool havePreviousId = false;
    bool found = false;
    for (usdGen::UsdGenDeviceTileMetadata const &candidate : geometry.tiles) {
        // Validate the complete partition, not just the requested range: a
        // malformed sibling can otherwise make a generation's tile layout
        // ambiguous even when this candidate happens to be in bounds.
        if ((havePreviousId && candidate.tile <= previousId) ||
            candidate.elementCount > candidate.pointCount ||
            (candidate.elementCount == 0 && candidate.pointCount != 0) ||
            candidate.firstElement != curveEnd || candidate.firstPoint != pointEnd ||
            candidate.firstElement > geometry.elementCount ||
            candidate.elementCount > geometry.elementCount - candidate.firstElement ||
            candidate.firstPoint > geometry.pointCount ||
            candidate.pointCount > geometry.pointCount - candidate.firstPoint ||
            candidate.firstPoint > std::numeric_limits<uint32_t>::max() ||
            candidate.pointCount >
                std::numeric_limits<uint32_t>::max() - candidate.firstPoint) {
            return false;
        }
        curveEnd += candidate.elementCount;
        pointEnd += candidate.pointCount;
        previousId = candidate.tile;
        havePreviousId = true;
        if (candidate.tile == *_tileId) {
            *tile = candidate;
            found = true;
        }
    }
    return found && curveEnd == geometry.elementCount &&
        pointEnd == geometry.pointCount;
}

bool
UsdGenCudaGlComputation::_Describe(
    HdTupleType *tuple, size_t *count, size_t *elementBytes) const noexcept
{
    if (_destinationName.IsEmpty() || !_generation ||
        std::strcmp(usdGen::UsdGenBackendName(_generation->Identity().backend), "cuda") != 0 ||
        _generation->Identity().deviceIndex < 0 || !tuple || !count ||
        !elementBytes) {
        return false;
    }

    uint64_t channelCount = 0;
    if (_tileId) {
        usdGen::UsdGenDeviceTileMetadata tile;
        if (!_SelectedTile(&tile)) {
            return false;
        }
        using S = usdGen::UsdGenDeviceChannelSemantic;
        switch (_semantic) {
        case S::Points:
        case S::RestPoints:
        case S::Widths:
        case S::HairT:
            channelCount = tile.pointCount;
            break;
        case S::ElementOffsets:
            if (tile.elementCount == std::numeric_limits<uint64_t>::max()) {
                return false;
            }
            channelCount = tile.elementCount + 1;
            break;
        case S::StableIds:
        case S::RootPrim:
        case S::RootUV:
            channelCount = tile.elementCount;
            break;
        default:
            return false;
        }
    } else if (!_WholeChannelCount(&channelCount)) {
        return false;
    }
    if (!_ToSize(channelCount, count)) {
        return false;
    }

    using S = usdGen::UsdGenDeviceChannelSemantic;
    switch (_semantic) {
    case S::Points:
    case S::RestPoints:
        *tuple = HdTupleType{HdTypeFloatVec3, 1};
        *elementBytes = sizeof(float) * 3;
        return true;
    case S::Widths:
    case S::HairT:
        *tuple = HdTupleType{HdTypeFloat, 1};
        *elementBytes = sizeof(float);
        return true;
    case S::ElementOffsets:
        *tuple = HdTupleType{HdTypeUInt32, 1};
        *elementBytes = sizeof(uint32_t);
        return true;
    case S::StableIds:
        // HdTypeUInt32Vec2 preserves each uint64 id as its exact two words.
        *tuple = HdTupleType{HdTypeUInt32Vec2, 1};
        *elementBytes = sizeof(uint64_t);
        return true;
    case S::RootPrim:
        *tuple = HdTupleType{HdTypeInt32, 1};
        *elementBytes = sizeof(int32_t);
        return true;
    case S::RootUV:
        *tuple = HdTupleType{HdTypeFloatVec2, 1};
        *elementBytes = sizeof(float) * 2;
        return true;
    default:
        return false;
    }
}

int
UsdGenCudaGlComputation::GetNumOutputElements() const
{
    HdTupleType tuple;
    size_t count = 0, elementBytes = 0;
    return _Describe(&tuple, &count, &elementBytes) && _FitsInt(count)
        ? static_cast<int>(count) : 0;
}

void
UsdGenCudaGlComputation::GetBufferSpecs(HdBufferSpecVector *specs) const
{
    if (!specs) {
        return;
    }
    HdTupleType tuple;
    size_t count = 0, elementBytes = 0;
    if (_Describe(&tuple, &count, &elementBytes) && _FitsInt(count)) {
        specs->emplace_back(_destinationName, tuple);
    }
}

void
UsdGenCudaGlComputation::Execute(
    HdBufferArrayRangeSharedPtr const &range,
    HdResourceRegistry *)
{
    _succeeded = false;
    _error.clear();
    if (usdGen::UsdGenExecutionPipeline::IsExecuting()) {
        _Fail("CUDA-GL transfer requires an external graphics commit thread");
        return;
    }

    HdTupleType expectedTuple;
    size_t count = 0, elementBytes = 0;
    if (!_Describe(&expectedTuple, &count, &elementBytes) || !_FitsInt(count)) {
        _Fail("invalid CUDA generation or unsupported channel");
        return;
    }
    using S = usdGen::UsdGenDeviceChannelSemantic;

    auto const destinationRange =
        std::dynamic_pointer_cast<HdStBufferArrayRange>(range);
    if (!destinationRange) {
        _Fail("destination is not an HdStBufferArrayRange");
        return;
    }
    HdStBufferResourceSharedPtr const destination =
        destinationRange->GetResource(_destinationName);
    if (!destination || destination->GetTupleType() != expectedTuple ||
        destinationRange->GetNumElements() != count) {
        _Fail("destination tuple or element count does not match CUDA channel");
        return;
    }
    usdGen::UsdGenDeviceValueType expectedValueType =
        usdGen::UsdGenDeviceValueType::Float32;
    uint32_t expectedArity = 1;
    using VT = usdGen::UsdGenDeviceValueType;
    switch (_semantic) {
    case S::Points:
    case S::RestPoints: expectedValueType = VT::Float32x3; expectedArity = 3; break;
    case S::RootUV: expectedValueType = VT::Float32x2; expectedArity = 2; break;
    case S::ElementOffsets: expectedValueType = VT::UInt32; break;
    case S::StableIds: expectedValueType = VT::UInt64; break;
    case S::RootPrim: expectedValueType = VT::Int32; break;
    case S::Widths:
    case S::HairT: break;
    default: _Fail("unsupported CUDA channel semantic"); return;
    }
    uint64_t metadataCount = 0;
    if (!_WholeChannelCount(&metadataCount)) {
        _Fail("CUDA channel metadata has an invalid whole-generation cardinality");
        return;
    }
    bool metadataMatches = false;
    for (usdGen::UsdGenDeviceChannelMetadata const &channel :
         _generation->Channels()) {
        // Channel metadata always describes the entire immutable generation;
        // `count` above intentionally describes only this destination tile.
        if (channel.semantic == _semantic && channel.elementCount == metadataCount &&
            channel.type == expectedValueType && channel.arity == expectedArity &&
            channel.strideBytes == elementBytes) {
            metadataMatches = true;
            break;
        }
    }
    if (!metadataMatches) {
        _Fail("CUDA channel metadata does not match its required Storm layout");
        return;
    }
    int glDevices[16] = {};
    unsigned int glDeviceCount = 0;
    cudaError_t status = cudaGLGetDevices(
        &glDeviceCount, glDevices, 16, cudaGLDeviceListAll);
    if (status != cudaSuccess) {
        _Fail("cudaGLGetDevices", status);
        return;
    }
    bool const glMatchesGeneration = [&] {
        for (unsigned int i = 0; i != glDeviceCount; ++i) {
            if (glDevices[i] == _generation->Identity().deviceIndex) return true;
        }
        return false;
    }();
    if (!glMatchesGeneration) {
        _Fail("current GL context is not associated with the generation CUDA device");
        return;
    }
    int previousDevice = -1;
    if ((status = cudaGetDevice(&previousDevice)) != cudaSuccess) {
        _Fail("cudaGetDevice", status);
        return;
    }
    if ((status = cudaSetDevice(_generation->Identity().deviceIndex)) != cudaSuccess) {
        _Fail("cudaSetDevice", status);
        return;
    }
    auto const restoreDevice = [&] {
        return previousDevice != _generation->Identity().deviceIndex
            ? cudaSetDevice(previousDevice) : cudaSuccess;
    };

    cudaStream_t stream = nullptr;
    if ((status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)) != cudaSuccess) {
        restoreDevice();
        _Fail("cudaStreamCreateWithFlags", status);
        return;
    }

    // The lease keeps the generation's storage alive and orders its producer
    // before this stream; hold it until GL is unmapped and the stream fenced.
    usdGen::UsdGenDeviceLease lease = _generation->AcquireLease(
        reinterpret_cast<usdGen::UsdGenDeviceStream>(stream));
    if (lease && lease.WaitUntilReady() != usdGen::UsdGenDeviceStatus::Ok) {
        lease = {};
    }
    if (!lease) {
        cudaStreamDestroy(stream);
        restoreDevice();
        _Fail(_tileId ? "CUDA tile generation rejected the consumer stream" :
                        "CUDA generation rejected the consumer stream");
        return;
    }

    // A tile slice keeps global offset values; every other channel is
    // tile-local.
    usdGen::UsdGenDeviceChannelAddress address;
    void const *source = nullptr;
    size_t sourceCount = 0;
    if (_generation->ChannelAddress(_semantic, _tileId ? *_tileId : UINT32_MAX, &address) &&
        address.strideBytes == elementBytes && address.bytes % elementBytes == 0 &&
        _ToSize(address.bytes / elementBytes, &sourceCount)) {
        source = reinterpret_cast<void const *>(static_cast<uintptr_t>(address.address));
    } else {
        sourceCount = 0;
    }
    if (sourceCount != count || (count && !source)) {
        lease = {};
        cudaStreamDestroy(stream);
        restoreDevice();
        _Fail("CUDA channel shape does not match immutable generation metadata");
        return;
    }

    // A fresh empty-only Storm BAR is structurally valid but need not own a
    // backing Hgi allocation.  It has no CUDA geometry copy to submit; retain
    // the source lease through its completion fence before reporting Ready.
    if (count == 0) {
        cudaError_t const fenceStatus = cudaStreamSynchronize(stream);
        lease = {};
        cudaError_t const destroyStatus = cudaStreamDestroy(stream);
        cudaError_t const restoreStatus = restoreDevice();
        if (fenceStatus != cudaSuccess) {
            _Fail("cudaStreamSynchronize for empty CUDA-GL transfer", fenceStatus);
            return;
        }
        if (destroyStatus != cudaSuccess || restoreStatus != cudaSuccess) {
            _Fail("CUDA-GL empty transfer cleanup",
                  destroyStatus != cudaSuccess ? destroyStatus : restoreStatus);
            return;
        }
        _succeeded = true;
        return;
    }

    if (!destination->GetHandle()) {
        lease = {};
        cudaStreamDestroy(stream);
        restoreDevice();
        _Fail("destination has no allocated Hgi buffer");
        return;
    }
    HgiGLBuffer *const glBuffer = dynamic_cast<HgiGLBuffer *>(
        destination->GetHandle().Get());
    if (!glBuffer || !glBuffer->GetBufferId()) {
        lease = {};
        cudaStreamDestroy(stream);
        restoreDevice();
        _Fail("destination is not an HgiGLBuffer");
        return;
    }
    if (destination->GetOffset() < 0 || destination->GetStride() <= 0 ||
        destinationRange->GetByteOffset(_destinationName) < 0) {
        lease = {};
        cudaStreamDestroy(stream);
        restoreDevice();
        _Fail("destination has a negative Storm byte offset or stride");
        return;
    }

    size_t destinationOffset = 0;
    size_t span = 0;
    size_t destinationEnd = 0;
    if (!_Add(static_cast<size_t>(destinationRange->GetByteOffset(_destinationName)),
              static_cast<size_t>(destination->GetOffset()), &destinationOffset) ||
        static_cast<size_t>(destination->GetStride()) < elementBytes ||
        !_Mul(count - 1, static_cast<size_t>(destination->GetStride()), &span) ||
        !_Add(destinationOffset, span, &destinationEnd) ||
        !_Add(destinationEnd, elementBytes, &destinationEnd) ||
        destinationEnd > glBuffer->GetByteSizeOfResource()) {
        lease = {};
        cudaStreamDestroy(stream);
        restoreDevice();
        _Fail("destination Storm range overflows or exceeds its GL buffer");
        return;
    }

    cudaGraphicsResource_t graphics = nullptr;
    bool mapped = false;
    status = cudaGraphicsGLRegisterBuffer(
        &graphics, glBuffer->GetBufferId(), cudaGraphicsRegisterFlagsNone);
    if (status == cudaSuccess) {
        status = cudaGraphicsMapResources(1, &graphics, stream);
        mapped = status == cudaSuccess;
    }
    void *mappedDestination = nullptr;
    size_t mappedBytes = 0;
    if (status == cudaSuccess) {
        status = cudaGraphicsResourceGetMappedPointer(
            &mappedDestination, &mappedBytes, graphics);
    }
    if (status == cudaSuccess && mappedBytes < glBuffer->GetByteSizeOfResource()) {
        _Fail("mapped GL buffer is smaller than Storm's HgiGLBuffer allocation");
        status = cudaErrorInvalidValue;
    }
    if (status == cudaSuccess && count) {
        status = cudaMemcpy2DAsync(
            static_cast<unsigned char *>(mappedDestination) + destinationOffset,
            static_cast<size_t>(destination->GetStride()), source, elementBytes,
            elementBytes, count, cudaMemcpyDeviceToDevice, stream);
    }

    // Unmap is the CUDA-to-GL ownership hand-off.  Do not unregister if it
    // fails: CUDA's ownership is then unknown, so freeing the registration is
    // unsafe.  The process/context teardown is the recovery boundary.
    cudaError_t const workStatus = status;
    cudaError_t unmapStatus = cudaSuccess;
    if (mapped) {
        unmapStatus = cudaGraphicsUnmapResources(1, &graphics, stream);
    }
    if (unmapStatus != cudaSuccess) {
        _Fail("cudaGraphicsUnmapResources", unmapStatus);
        TF_FATAL_ERROR("usdGen CUDA-GL ownership hand-off failed: %s",
                       _error.c_str());
        std::terminate();
    }
    // cudaGraphicsUnmapResources is ordered on stream.  Check that stream
    // explicitly: the lease's Complete also fences it, but deliberately
    // cannot return its status to this computation.
    if ((status = cudaStreamSynchronize(stream)) != cudaSuccess) {
        _Fail("cudaStreamSynchronize after CUDA-GL unmap", status);
        TF_FATAL_ERROR("usdGen CUDA-GL ownership fence failed: %s",
                       _error.c_str());
        std::terminate();
    }
    // Completion fences the consumer stream, including the asynchronous
    // unmap, before the GL resource registration is released.
    lease = {};
    if (graphics) {
        cudaError_t const unregisterStatus = cudaGraphicsUnregisterResource(graphics);
        if (unregisterStatus != cudaSuccess) {
            _Fail("cudaGraphicsUnregisterResource", unregisterStatus);
            TF_FATAL_ERROR("usdGen CUDA-GL resource release failed: %s",
                           _error.c_str());
            std::terminate();
        }
    }

    // The source lease was completed while stream was valid; its owner
    // synchronized this stream before permitting storage reclamation.
    cudaError_t const destroyStatus = cudaStreamDestroy(stream);
    cudaError_t const restoreStatus = restoreDevice();
    if (workStatus != cudaSuccess) {
        _Fail("CUDA-to-GL copy", workStatus);
        return;
    }
    if (destroyStatus != cudaSuccess || restoreStatus != cudaSuccess) {
        _Fail("CUDA-GL stream/device cleanup", destroyStatus != cudaSuccess ? destroyStatus : restoreStatus);
        return;
    }
    _succeeded = true;
}

} // namespace usdGenImaging

#endif // USDGEN_HAS_CUDA_GL_INTEROP
