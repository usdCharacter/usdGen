// CUDA to Storm/OpenGL buffer computation bridge.  This is deliberately a
// commit-thread primitive: Execute requires the current thread to own the GL
// context used by Storm.  It neither publishes scene data nor schedules work.
#ifndef USDGEN_IMAGING_CUDA_GL_COMPUTATION_H
#define USDGEN_IMAGING_CUDA_GL_COMPUTATION_H

#include "pxr/pxr.h"

#ifdef USDGEN_HAS_CUDA_GL_INTEROP

#include "usdGen/deviceGeneration.h"

#include "pxr/imaging/hdSt/computation.h"
#include "pxr/imaging/hd/types.h"
#include "pxr/base/tf/token.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGenImaging {

/// Copies one immutable CUDA geometry channel directly into a Storm GL BAR.
/// The instance is single-context/single-thread: its caller must keep the
/// generation immutable and invoke Execute on Storm's GL commit context.  A
/// caller must not publish/use the destination BAR unless Succeeded() is true;
/// a submitted device copy has no rollback path.
/// This initial bridge fences on the graphics commit thread and registers
/// each transfer separately; it does not claim asynchronous frame throughput.
/// It must never execute on a CUDA host callback or an engine command owner.
class UsdGenCudaGlComputation final : public HdStComputation
{
public:
    UsdGenCudaGlComputation(
        std::shared_ptr<const usdGen::UsdGenDeviceGeneration> generation,
        usdGen::UsdGenDeviceChannelSemantic semantic,
        TfToken destinationName,
        std::optional<uint32_t> tileId = std::nullopt);

    void Execute(HdBufferArrayRangeSharedPtr const &range,
                 HdResourceRegistry *resourceRegistry) override;
    int GetNumOutputElements() const override;
    void GetBufferSpecs(HdBufferSpecVector *specs) const override;

    bool Succeeded() const noexcept { return _succeeded; }
    std::string const &Error() const noexcept { return _error; }

private:
    bool _Describe(HdTupleType *tuple, size_t *count,
                   size_t *elementBytes) const noexcept;
    bool _SelectedTile(usdGen::UsdGenDeviceTileMetadata *tile) const noexcept;
    bool _WholeChannelCount(uint64_t *count) const noexcept;
    void _Fail(char const *where, int cudaError) noexcept;
    void _Fail(std::string message) noexcept;

    std::shared_ptr<const usdGen::UsdGenDeviceGeneration> _generation;
    usdGen::UsdGenDeviceChannelSemantic _semantic;
    TfToken _destinationName;
    std::optional<uint32_t> _tileId;
    bool _succeeded = false;
    std::string _error;
};

} // namespace usdGenImaging

#endif // USDGEN_HAS_CUDA_GL_INTEROP
#endif // USDGEN_IMAGING_CUDA_GL_COMPUTATION_H
