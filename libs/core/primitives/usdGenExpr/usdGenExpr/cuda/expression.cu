#include "usdGenExpr/cuda/expression.h"
#include "usdGenCuda/gpu/cudaCompletionLane.h"
#include "usdGenCuda/gpu/cudaCompat.h"
// The interpreter itself lives in expressions/irExec.h and is compiled for the
// device here and for the host by expressions/cpuEvaluator.cpp. Do not add a
// second copy of any IR semantics to this file.
#include "usdGenExpr/expressions/irExec.h"

#include <climits>
#include <cmath>
#include <cstring>
#include <limits>
#include <cuda_fp16.h>

namespace usdGen::gpu {
namespace {
constexpr unsigned kRegisters = expr::kExprRegisters;
using expr::IROp;
using expr::Variable;
using expr::Domain;
using expr::ScalarType;
using expr::ValidProgram;

bool IsDomain(Domain d) { return expr::ExprIsDomain(d); }

ExpressionStatus ValidInputs(ExpressionInputs const& inputs, ExpressionOutput output,
                             unsigned outputCount) {
    if (!IsDomain(inputs.context.domain) || inputs.count > size_t(INT_MAX) ||
        inputs.count != output.count || (inputs.count && !output.data) ||
        output.type == ScalarType::Invalid || output.components != outputCount ||
        (inputs.context.domain == Domain::Groom && inputs.count != 1))
        return ExpressionStatus::InvalidArgument;
    for (auto const& field : inputs.fields) {
        if (!field.data && field.count == 0) continue;
        if (!field.data || !IsDomain(field.domain) || field.components == 0 ||
            field.components > 4 || field.count > size_t(INT_MAX))
            return ExpressionStatus::InvalidArgument;
        size_t expected = field.domain == Domain::Groom ? 1 : inputs.count;
        if (field.domain == Domain::Primitive && inputs.context.domain == Domain::Point) {
            expected = inputs.primitiveCount;
            if (!inputs.pointToPrimitive.data || inputs.pointToPrimitive.size != inputs.count)
                return ExpressionStatus::InvalidArgument;
        } else if (field.domain != Domain::Groom && field.domain != inputs.context.domain)
            return ExpressionStatus::InvalidArgument;
        if (field.count != expected) return ExpressionStatus::InvalidArgument;
    }
    return ExpressionStatus::Ok;
}

struct OutputRegisters { uint16_t value[4]; };
__global__ void Execute(expr::IRInstruction const* code, size_t count, OutputRegisters result,
                        ExpressionInputs inputs, ExpressionOutput output, int* error) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= inputs.count) return;
    double r[kRegisters];
    expr::ExecuteElement(code, count, r, inputs, i);
    // Pure predication: a dead conditional arm may carry NaN, but only the
    // selected result is validated/published. No instruction has side effects.
    for (unsigned c = 0; c < output.components; ++c)
        if (!expr::Store(r[result.value[c]], output, i * output.components + c))
            atomicExch(error, 1);
}
} // namespace

CudaExpressionProgram::~CudaExpressionProgram() {
    if (unprovenWork_) {
        // A fresh D2H terminal was never proved.  Do not make destruction
        // issue CUDA frees against work which may still be using these bytes.
        code_.quarantine(); error_.quarantine();
        freshHostErrorPermit_.Abandon(); freshHostCodePermit_.Abandon();
        freshHostError_ = nullptr; freshHostCode_ = nullptr; ready_ = nullptr;
        return;
    }
    const bool owns = ready_ || freshHostError_ || freshHostCode_ ||
        code_.size() || error_.size();
    if (!owns) return;
    int previous = -1;
    bool const selected = cudaGetDevice(&previous) == cudaSuccess && deviceIndex_ >= 0 &&
        cudaSetDevice(deviceIndex_) == cudaSuccess;
    auto quarantine = [&] {
        code_.quarantine(); error_.quarantine(); ready_ = nullptr;
        freshHostError_ = nullptr; freshHostCode_ = nullptr;
        freshHostErrorPermit_.Abandon(); freshHostCodePermit_.Abandon();
    };
    if (!selected || (ready_ && cudaEventSynchronize(ready_) != cudaSuccess)) {
        quarantine();
        if (selected && previous != deviceIndex_) cudaSetDevice(previous);
        return;
    }
    if (ready_) {
        if (cudaEventDestroy(ready_) != cudaSuccess) {
            quarantine();
            if (previous != deviceIndex_) cudaSetDevice(previous);
            return;
        }
        ready_ = nullptr;
    }
    code_.reset(0); error_.reset(0);
    if (freshHostError_) {
        if (cudaFreeHost(freshHostError_) == cudaSuccess) freshHostErrorPermit_.Release();
        else freshHostErrorPermit_.Abandon();
        freshHostError_ = nullptr;
    }
    if (freshHostCode_) {
        if (cudaFreeHost(freshHostCode_) == cudaSuccess) freshHostCodePermit_.Release();
        else freshHostCodePermit_.Abandon();
        freshHostCode_ = nullptr;
    }
    if (previous != deviceIndex_) cudaSetDevice(previous);
}

ExpressionStatus CudaExpressionProgram::Upload(expr::IRProgram const& program, cudaStream_t stream,
                                                UsdGenExecutionMemoryReservation* reservation,
                                                UsdGenExecutionResourceKind kind) {
    if (pending_ || freshUploadPending_ || unprovenWork_ || freshPoisoned_) return ExpressionStatus::InvalidArgument;
    if (!ValidProgram(program)) return ExpressionStatus::InvalidProgram;
    int current = -1, streamDevice = -1;
    if (cudaGetDevice(&current) != cudaSuccess ||
        (stream && (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess || streamDevice != current)))
        return ExpressionStatus::CudaError;
    if (deviceIndex_ >= 0 && deviceIndex_ != current) return ExpressionStatus::InvalidArgument;
    deviceIndex_ = current;
    instructionCount_ = 0;
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return ExpressionStatus::CudaError;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        code_.reset(program.instructions.size() * sizeof(expr::IRInstruction), reservation, kind) != cudaSuccess ||
        error_.reset(1, reservation, kind) != cudaSuccess) return ExpressionStatus::CudaError;
    if (cudaMemcpyAsync(code_.data(), program.instructions.data(), code_.size(),
                        cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaEventRecord(ready_, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess) return ExpressionStatus::CudaError;
    // The caller's host IR need not outlive Upload.
    instructionCount_ = program.instructions.size(); result_ = program.result;
    outputCount_ = program.outputCount ? program.outputCount : 1;
    for (unsigned c = 0; c < outputCount_; ++c)
        output_[c] = program.outputCount ? program.output[c] : program.result;
    return ExpressionStatus::Ok;
}

ExpressionStatus CudaExpressionProgram::Evaluate(ExpressionInputs const& inputs,
                                                ExpressionOutput output, cudaStream_t stream) {
    if (pending_ || freshUploadPending_ || unprovenWork_ || !instructionCount_)
        return ExpressionStatus::InvalidArgument;
    ExpressionStatus const valid = ValidInputs(inputs, output, outputCount_);
    if (valid != ExpressionStatus::Ok) return valid;
    int current = -1, streamDevice = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != deviceIndex_ ||
        (stream && (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess ||
                    streamDevice != deviceIndex_))) return ExpressionStatus::CudaError;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess)
        return ExpressionStatus::CudaError;
    if (inputs.count) {
        OutputRegisters outputs{};
        for (unsigned c = 0; c < outputCount_; ++c) outputs.value[c] = output_[c];
        Execute<<<(inputs.count + 127) / 128, 128, 0, stream>>>(
            reinterpret_cast<expr::IRInstruction const*>(code_.data()), instructionCount_,
            outputs, inputs, output, error_.data());
        if (cudaGetLastError() != cudaSuccess) return ExpressionStatus::CudaError;
    }
    if (cudaEventRecord(ready_, stream) != cudaSuccess) return ExpressionStatus::CudaError;
    pending_ = true;
    return ExpressionStatus::Ok;
}

ExpressionStatus CudaExpressionProgram::Finish(cudaStream_t stream) {
    if (!pending_ || freshUploadPending_ || unprovenWork_) return ExpressionStatus::InvalidArgument;
    int current = -1, streamDevice = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != deviceIndex_ ||
        (stream && (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess ||
                    streamDevice != deviceIndex_))) return ExpressionStatus::CudaError;
    int error = 0;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(&error, error_.data(), sizeof(error), cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
        CudaFenceStream(stream) != cudaSuccess) return ExpressionStatus::CudaError;
    pending_ = false;
    return error ? ExpressionStatus::InvalidValue : ExpressionStatus::Ok;
}

ExpressionStatus CudaExpressionProgram::UploadFresh(expr::IRProgram const& program,
                                                     cudaStream_t stream,
                                                     UsdGenExecutionMemoryReservation* reservation,
                                                     UsdGenExecutionResourceKind kind) {
    if (pending_ || freshUploadPending_ || unprovenWork_ || freshPoisoned_ || !ValidProgram(program))
        return pending_ || freshUploadPending_ || unprovenWork_ || freshPoisoned_
            ? ExpressionStatus::InvalidArgument : ExpressionStatus::InvalidProgram;
    // Capture is rejected before stream-device inspection or allocation.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return ExpressionStatus::InvalidArgument;
    int current = -1, streamDevice = -1;
    if (cudaGetDevice(&current) != cudaSuccess ||
        (stream && (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess ||
                    streamDevice != current)) ||
        (deviceIndex_ >= 0 && deviceIndex_ != current)) return ExpressionStatus::CudaError;
    // Record ownership before any allocation so a partial fresh setup still
    // has a safe destruction device.
    deviceIndex_ = current;
    if (!ready_ && cudaEventCreateWithFlags(&ready_, cudaEventDisableTiming) != cudaSuccess)
        return ExpressionStatus::CudaError;
    if (code_.reset(program.instructions.size() * sizeof(expr::IRInstruction), reservation, kind) != cudaSuccess ||
        error_.reset(1, reservation, kind) != cudaSuccess) return ExpressionStatus::CudaError;
    if (freshHostError_) {
        if (cudaFreeHost(freshHostError_) != cudaSuccess) {
            freshHostErrorPermit_.Abandon(); freshHostError_ = nullptr; freshPoisoned_ = true;
            return ExpressionStatus::CudaError;
        }
        freshHostErrorPermit_.Release(); freshHostError_ = nullptr;
    }
    if (freshHostCode_) {
        if (cudaFreeHost(freshHostCode_) != cudaSuccess) {
            freshHostCodePermit_.Abandon(); freshHostCode_ = nullptr; freshPoisoned_ = true;
            return ExpressionStatus::CudaError;
        }
        freshHostCodePermit_.Release(); freshHostCode_ = nullptr;
    }
    auto errorPermit = TryReserveCudaExecutionBytes(sizeof(int), kind, reservation);
    auto codePermit = TryReserveCudaExecutionBytes(code_.size(), kind, reservation);
    int* nextError = nullptr;
    expr::IRInstruction* nextCode = nullptr;
    bool hostFreeFailed = false;
    if (!errorPermit || !codePermit ||
        cudaHostAlloc(reinterpret_cast<void**>(&nextError), sizeof(int), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc(reinterpret_cast<void**>(&nextCode), code_.size(), cudaHostAllocDefault) != cudaSuccess) {
        if (nextError) {
            if (cudaFreeHost(nextError) == cudaSuccess) errorPermit->Release();
            else { errorPermit->Abandon(); hostFreeFailed = true; }
        }
        if (nextCode) {
            if (cudaFreeHost(nextCode) == cudaSuccess) codePermit->Release();
            else { codePermit->Abandon(); hostFreeFailed = true; }
        }
        if (hostFreeFailed) freshPoisoned_ = true;
        return ExpressionStatus::CudaError;
    }
    freshHostError_ = nextError;
    freshHostCode_ = nextCode;
    freshHostErrorPermit_ = std::move(*errorPermit);
    freshHostCodePermit_ = std::move(*codePermit);
    freshHostCodeBytes_ = code_.size();
    std::memcpy(freshHostCode_, program.instructions.data(), freshHostCodeBytes_);
    *freshHostError_ = std::numeric_limits<int>::min();
    // From this point the caller may have no proof whether DMA consumed the
    // owned pinned bytes, so every error path preserves them conservatively.
    unprovenWork_ = true; freshFailed_ = false; freshEvaluated_ = false;
    freshStatusEnqueued_ = false;
    if (cudaMemcpyAsync(code_.data(), freshHostCode_, freshHostCodeBytes_, cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaEventRecord(ready_, stream) != cudaSuccess) {
        freshFailed_ = true;
        return ExpressionStatus::CudaError;
    }
    instructionCount_ = program.instructions.size(); result_ = program.result;
    outputCount_ = program.outputCount ? program.outputCount : 1;
    for (unsigned c = 0; c < outputCount_; ++c)
        output_[c] = program.outputCount ? program.output[c] : program.result;
    freshUploadPending_ = true;
    return ExpressionStatus::Ok;
}

ExpressionStatus CudaExpressionProgram::EvaluateFresh(ExpressionInputs const& inputs,
                                                       ExpressionOutput output,
                                                       cudaStream_t stream) {
    if (!freshUploadPending_ || freshEvaluated_ || freshStatusEnqueued_ || freshFailed_ ||
        !unprovenWork_) return ExpressionStatus::InvalidArgument;
    ExpressionStatus const valid = ValidInputs(inputs, output, outputCount_);
    if (valid != ExpressionStatus::Ok) return valid;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess ||
        capture != cudaStreamCaptureStatusNone) return ExpressionStatus::InvalidArgument;
    int current = -1, streamDevice = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != deviceIndex_ ||
        (stream && (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess ||
                    streamDevice != deviceIndex_))) return ExpressionStatus::CudaError;
    if (cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemsetAsync(error_.data(), 0, sizeof(int), stream) != cudaSuccess) {
        freshFailed_ = true; return ExpressionStatus::CudaError;
    }
    if (inputs.count) {
        OutputRegisters outputs{};
        for (unsigned c = 0; c < outputCount_; ++c) outputs.value[c] = output_[c];
        Execute<<<(inputs.count + 127) / 128, 128, 0, stream>>>(
            reinterpret_cast<expr::IRInstruction const*>(code_.data()), instructionCount_,
            outputs, inputs, output, error_.data());
        if (cudaGetLastError() != cudaSuccess) { freshFailed_ = true; return ExpressionStatus::CudaError; }
    }
    if (cudaEventRecord(ready_, stream) != cudaSuccess) { freshFailed_ = true; return ExpressionStatus::CudaError; }
    pending_ = true; freshEvaluated_ = true;
    return ExpressionStatus::Ok;
}

ExpressionStatus CudaExpressionProgram::EnqueueFreshStatus(cudaStream_t stream) {
    if (!pending_ || !freshUploadPending_ || !freshEvaluated_ || freshStatusEnqueued_ ||
        freshFailed_ || !freshHostError_) return ExpressionStatus::InvalidArgument;
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &capture) != cudaSuccess || capture != cudaStreamCaptureStatusNone) {
        freshFailed_ = true; return ExpressionStatus::InvalidArgument;
    }
    int current = -1, streamDevice = -1;
    if (cudaGetDevice(&current) != cudaSuccess || current != deviceIndex_ ||
        (stream && (cudaStreamGetDevice(stream, &streamDevice) != cudaSuccess || streamDevice != deviceIndex_)) ||
        cudaStreamWaitEvent(stream, ready_, 0) != cudaSuccess ||
        cudaMemcpyAsync(freshHostError_, error_.data(), sizeof(int), cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        freshFailed_ = true; return ExpressionStatus::CudaError;
    }
    freshStatusEnqueued_ = true;
    return ExpressionStatus::Ok;
}

ExpressionStatus CudaExpressionProgram::CommitFreshFinish() {
    if (!pending_ || !freshUploadPending_ || !freshEvaluated_ || !freshStatusEnqueued_ ||
        freshFailed_ || !freshHostError_ || *freshHostError_ == std::numeric_limits<int>::min())
        return ExpressionStatus::InvalidArgument;
    int const error = *freshHostError_;
    pending_ = false; freshUploadPending_ = false; freshEvaluated_ = false;
    freshStatusEnqueued_ = false; freshFailed_ = false; unprovenWork_ = false;
    return error ? ExpressionStatus::InvalidValue : ExpressionStatus::Ok;
}
} // namespace usdGen::gpu
