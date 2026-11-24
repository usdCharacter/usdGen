// Runtime-API shims for CUDA toolkits older than the ones the sources target.
#ifndef USDGEN_GPU_CUDA_COMPAT_H
#define USDGEN_GPU_CUDA_COMPAT_H

#include <cuda_runtime.h>

#if CUDART_VERSION < 12080
#include <cuda.h>

// cudaStreamGetDevice arrived with CUDA 12.8. Older runtimes reach the same
// answer through the driver API: a stream belongs to exactly one context and
// a context to exactly one device. The push/pop pair leaves the caller's
// current context untouched. Like the runtime original this is not a
// capture-safe query, so callers rule out capture first.
inline cudaError_t cudaStreamGetDevice(cudaStream_t stream, int* device)
{
    if (!device) return cudaErrorInvalidValue;
    if (!stream) return cudaGetDevice(device);
    CUcontext context = nullptr;
    if (cuStreamGetCtx(stream, &context) != CUDA_SUCCESS || !context)
        return cudaErrorInvalidResourceHandle;
    if (cuCtxPushCurrent(context) != CUDA_SUCCESS)
        return cudaErrorInvalidResourceHandle;
    CUdevice ordinal = -1;
    CUresult const result = cuCtxGetDevice(&ordinal);
    CUcontext popped = nullptr;
    cuCtxPopCurrent(&popped);
    if (result != CUDA_SUCCESS) return cudaErrorInvalidResourceHandle;
    *device = static_cast<int>(ordinal);
    return cudaSuccess;
}
#endif  // CUDART_VERSION < 12080

#endif  // USDGEN_GPU_CUDA_COMPAT_H
