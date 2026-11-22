// usdGen engine — internal operator registry (03 §8.2; ADR §3: internal in v1/v2).
//
// Maps a type name to a factory. There is exactly one kernel per type: the
// schema carries no algorithm-version knob, so nothing selects between
// alternative implementations of the same operator.
#ifndef USDGEN_OP_REGISTRY_H
#define USDGEN_OP_REGISTRY_H

#include "usdGen/op.h"

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

class UsdGenOpRegistry
{
public:
    using Factory = std::function<std::unique_ptr<UsdGenOp>()>;

    static UsdGenOpRegistry &Get();

    /// Register the kernel for `type`. The registry owns the factory, not the
    /// instances. Returns false on a duplicate type.
    /// Startup-only: fails after Freeze(). Libraries register through
    /// usdGenRegister (registrar.h), never by calling this directly.
    bool Register(TfToken const &type, Factory factory);

    /// Make the registry immutable. Readers then need no synchronisation.
    void Freeze() { _frozen.store(true, std::memory_order_release); }
    bool IsFrozen() const { return _frozen.load(std::memory_order_acquire); }

    /// Create an operator instance. nullptr when the type has no kernel
    /// (published types with unimplemented kernels: 02 §8.2).
    std::unique_ptr<UsdGenOp> Create(TfToken const &type) const;

    bool HasKernel(TfToken const &type) const;
    /// Native per-frame dependency declared by the registered kernel.
    bool ReadsTime(TfToken const &type) const;
    bool SamplesFrameInputs(TfToken const &type) const;
    /// Read the complete static operator contract without allocating an
    /// operator during graph compilation. The metadata is captured from the
    /// registration-time probe.
    bool GetOperatorContract(TfToken const &type,
                             size_t *outGeometryInputArity,
                             size_t *outReferenceInputArity,
                             UsdGenRole *outRole) const;
    /// Read the geometry-input contract without allocating an operator during
    /// graph compilation. Returns false for an unknown type.
    bool GetGeometryInputArity(TfToken const &type, size_t *outArity) const;

    /// All registered type names (sorted); used by diagnostics.
    std::vector<TfToken> KnownTypes() const;

private:
    UsdGenOpRegistry();
    struct Entry {
        TfToken type;
        Factory factory;
        size_t geometryInputArity;
        size_t referenceInputArity;
        UsdGenRole role;
        bool readsTime;
        bool samplesFrameInputs;
    };
    std::vector<Entry> _entries;
    std::atomic<bool> _frozen{false};
};

// --- M1 kernel factories (registered by usdGenOpRegistryInit) ----------------
std::unique_ptr<UsdGenOp> CreateScatterOp();
std::unique_ptr<UsdGenOp> CreateGrowOp();
std::unique_ptr<UsdGenOp> CreateNoiseOp();
std::unique_ptr<UsdGenOp> CreateLengthOp();
std::unique_ptr<UsdGenOp> CreateWidthOp();
std::unique_ptr<UsdGenOp> CreateWidthBlendOp();
std::unique_ptr<UsdGenOp> CreateReferenceSourceOp();
std::unique_ptr<UsdGenOp> CreateClumpOp();
std::unique_ptr<UsdGenOp> CreateGuideInterpolateOp();
std::unique_ptr<UsdGenOp> CreateCurlOp();
std::unique_ptr<UsdGenOp> CreateBendOp();
std::unique_ptr<UsdGenOp> CreateWaveOp();
std::unique_ptr<UsdGenOp> CreatePartOp();
std::unique_ptr<UsdGenOp> CreateDirectionOp();
std::unique_ptr<UsdGenOp> CreateSmoothOp();
std::unique_ptr<UsdGenOp> CreateResampleOp();
std::unique_ptr<UsdGenOp> CreateScaleOp();
std::unique_ptr<UsdGenOp> CreateStraightenOp();
std::unique_ptr<UsdGenOp> CreateDisplaceOp();
std::unique_ptr<UsdGenOp> CreateSculptLayerOp();
std::unique_ptr<UsdGenOp> CreateWindOp();
std::unique_ptr<UsdGenOp> CreateExprOp();
std::unique_ptr<UsdGenOp> CreateFreezeOp();
std::unique_ptr<UsdGenOp> CreateCollideOp();

/// Compatibility initializer. Built-ins are installed by registry construction;
/// concurrent compiler calls only obtain that completed immutable initial state.
void usdGenRegisterM1Operators();

}  // namespace usdGen

#endif  // USDGEN_OP_REGISTRY_H
