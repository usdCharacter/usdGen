// usdGen engine — operator registry implementation (M1 scaffold).
//
// M1 registers nine kernels (11-roadmap.md §2.2): UsdGenScatter,
// UsdGenGrow, UsdGenNoise, UsdGenLength, UsdGenWidth, UsdGenWidthBlend,
// UsdGenCurveSource, UsdGenDeform and UsdGenReferenceSource, plus the
// CPU-lane stylers UsdGenClump, UsdGenGuideInterpolate, UsdGenCurl,
// UsdGenBend, UsdGenWave, UsdGenPart, UsdGenDirection, UsdGenSmooth,
// UsdGenResample, UsdGenScale, UsdGenStraighten, UsdGenDisplace,
// UsdGenSculptLayer, UsdGenWind, UsdGenExprOp, UsdGenFreeze and
// UsdGenCollide. Later
// milestones register more via the same entry point; the registry is
// internal in v1/v2 (03 §8.2).
#include "usdGen/opRegistry.h"
#include "usdGen/registrar.h"

#include "usdGen/ops/scatter.h"
#include "usdGen/ops/grow.h"
#include "usdGen/ops/noise.h"
#include "usdGen/ops/length.h"
#include "usdGen/ops/width.h"
#include "usdGen/ops/widthBlend.h"
#include "usdGen/ops/curveSource.h"
#include "usdGen/ops/deform.h"
#include "usdGen/ops/referenceSource.h"
#include "usdGen/ops/clump.h"
#include "usdGen/ops/guideInterpolate.h"
#include "usdGen/ops/curl.h"
#include "usdGen/ops/bend.h"
#include "usdGen/ops/wave.h"
#include "usdGen/ops/part.h"
#include "usdGen/ops/direction.h"
#include "usdGen/ops/smooth.h"
#include "usdGen/ops/resample.h"
#include "usdGen/ops/scale.h"
#include "usdGen/ops/straighten.h"
#include "usdGen/ops/displace.h"
#include "usdGen/ops/sculptLayer.h"
#include "usdGen/ops/wind.h"
#include "usdGen/ops/exprOp.h"
#include "usdGen/ops/freeze.h"
#include "usdGen/ops/collide.h"
#include "usdGen/ops/braid.h"
#include "usdGen/ops/twist.h"
#include "usdGen/ops/trim.h"
#include "usdGen/ops/frizz.h"
#include "usdGen/ops/tieClump.h"
#include "usdGen/ops/cardSource.h"

#include "pxr/pxr.h"

PXR_NAMESPACE_USING_DIRECTIVE

namespace usdGen {

UsdGenOpRegistry &UsdGenOpRegistry::Get()
{
    static UsdGenOpRegistry instance;
    return instance;
}

bool UsdGenOpRegistry::Register(TfToken const &type, Factory factory)
{
    if (IsFrozen()) return false;
    for (Entry const &e : _entries) {
        if (e.type == type) return false;
    }
    if (!factory) return false;
    std::unique_ptr<UsdGenOp> probe = factory();
    if (!probe) return false;
    _entries.push_back(Entry{type, std::move(factory),
                             probe->GeometryInputArity(),
                             probe->ReferenceInputs().size(), probe->Role(),
                             probe->ReadsTime(), probe->SamplesFrameInputs()});
    return true;
}

std::unique_ptr<UsdGenOp> UsdGenOpRegistry::Create(TfToken const &type) const
{
    for (Entry const &e : _entries) {
        if (e.type == type && e.factory) return e.factory();
    }
    return nullptr;
}

std::vector<TfToken> UsdGenOpRegistry::KnownTypes() const
{
    std::vector<TfToken> types;
    for (Entry const &e : _entries) types.push_back(e.type);
    return types;
}

bool UsdGenOpRegistry::HasKernel(TfToken const &type) const
{
    for (auto const& entry : _entries)
        if (entry.type == type && entry.factory) return true;
    return false;
}

bool UsdGenOpRegistry::ReadsTime(TfToken const &type) const
{
    for (Entry const &entry : _entries)
        if (entry.type == type) return entry.readsTime;
    return false;
}

bool UsdGenOpRegistry::SamplesFrameInputs(TfToken const &type) const
{
    for (Entry const &entry : _entries)
        if (entry.type == type) return entry.samplesFrameInputs;
    return false;
}

bool UsdGenOpRegistry::GetGeometryInputArity(
    TfToken const &type, size_t *outArity) const
{
    return GetOperatorContract(type, outArity, nullptr, nullptr);
}

bool UsdGenOpRegistry::GetOperatorContract(
    TfToken const &type, size_t *outGeometryInputArity,
    size_t *outReferenceInputArity, UsdGenRole *outRole) const
{
    Entry const *best = nullptr;
    for (Entry const &entry : _entries) {
        if (entry.type != type) continue;
        best = &entry;
        break;
    }
    if (!best) return false;
    if (outGeometryInputArity)
        *outGeometryInputArity = best->geometryInputArity;
    if (outReferenceInputArity)
        *outReferenceInputArity = best->referenceInputArity;
    if (outRole) *outRole = best->role;
    return true;
}

std::unique_ptr<UsdGenOp> CreateScatterOp() { return std::make_unique<UsdGenScatterOp>(); }
std::unique_ptr<UsdGenOp> CreateGrowOp()    { return std::make_unique<UsdGenGrowOp>(); }
std::unique_ptr<UsdGenOp> CreateNoiseOp()   { return std::make_unique<UsdGenNoiseOp>(); }
std::unique_ptr<UsdGenOp> CreateLengthOp()  { return std::make_unique<UsdGenLengthOp>(); }
std::unique_ptr<UsdGenOp> CreateWidthOp()   { return std::make_unique<UsdGenWidthOp>(); }
std::unique_ptr<UsdGenOp> CreateWidthBlendOp() { return std::make_unique<UsdGenWidthBlendOp>(); }
std::unique_ptr<UsdGenOp> CreateCurveSourceOp() { return std::make_unique<UsdGenCurveSourceOp>(); }
std::unique_ptr<UsdGenOp> CreateDeformOp()    { return std::make_unique<UsdGenDeformOp>(); }
std::unique_ptr<UsdGenOp> CreateReferenceSourceOp() { return std::make_unique<UsdGenReferenceSourceOp>(); }
std::unique_ptr<UsdGenOp> CreateClumpOp() { return std::make_unique<UsdGenClumpOp>(); }
std::unique_ptr<UsdGenOp> CreateGuideInterpolateOp() { return std::make_unique<UsdGenGuideInterpolateOp>(); }
std::unique_ptr<UsdGenOp> CreateCurlOp() { return std::make_unique<UsdGenCurlOp>(); }
std::unique_ptr<UsdGenOp> CreateBendOp() { return std::make_unique<UsdGenBendOp>(); }
std::unique_ptr<UsdGenOp> CreateWaveOp() { return std::make_unique<UsdGenWaveOp>(); }
std::unique_ptr<UsdGenOp> CreatePartOp() { return std::make_unique<UsdGenPartOp>(); }
std::unique_ptr<UsdGenOp> CreateDirectionOp() { return std::make_unique<UsdGenDirectionOp>(); }
std::unique_ptr<UsdGenOp> CreateSmoothOp() { return std::make_unique<UsdGenSmoothOp>(); }
std::unique_ptr<UsdGenOp> CreateResampleOp() { return std::make_unique<UsdGenResampleOp>(); }
std::unique_ptr<UsdGenOp> CreateScaleOp() { return std::make_unique<UsdGenScaleOp>(); }
std::unique_ptr<UsdGenOp> CreateStraightenOp() { return std::make_unique<UsdGenStraightenOp>(); }
std::unique_ptr<UsdGenOp> CreateDisplaceOp() { return std::make_unique<UsdGenDisplaceOp>(); }
std::unique_ptr<UsdGenOp> CreateSculptLayerOp() { return std::make_unique<UsdGenSculptLayerOp>(); }
std::unique_ptr<UsdGenOp> CreateWindOp() { return std::make_unique<UsdGenWindOp>(); }
std::unique_ptr<UsdGenOp> CreateExprOp() { return std::make_unique<UsdGenExprOp>(); }
std::unique_ptr<UsdGenOp> CreateFreezeOp() { return std::make_unique<UsdGenFreezeOp>(); }
std::unique_ptr<UsdGenOp> CreateCollideOp() { return std::make_unique<UsdGenCollideOp>(); }
std::unique_ptr<UsdGenOp> CreateBraidOp() { return std::make_unique<UsdGenBraidOp>(); }
std::unique_ptr<UsdGenOp> CreateTwistOp() { return std::make_unique<UsdGenTwistOp>(); }
std::unique_ptr<UsdGenOp> CreateTrimOp() { return std::make_unique<UsdGenTrimOp>(); }
std::unique_ptr<UsdGenOp> CreateFrizzOp() { return std::make_unique<UsdGenFrizzOp>(); }
std::unique_ptr<UsdGenOp> CreateTieClumpOp() { return std::make_unique<UsdGenTieClumpOp>(); }
std::unique_ptr<UsdGenOp> CreateCardSourceOp() { return std::make_unique<UsdGenCardSourceOp>(); }

UsdGenOpRegistry::UsdGenOpRegistry()
{
    _entries.reserve(32);
    UsdGenRegistrar r(*this);
    usdGenRegister_usdGenCurves(r);
}

void usdGenRegisterM1Operators() { (void)UsdGenOpRegistry::Get(); }

}  // namespace usdGen

using namespace usdGen;

// Built-in curve operators. Moves to libs/core/primitives/usdGenCurves with
// step 6; until then it lives beside the registry and the constructor calls it.
extern "C" void usdGenRegister_usdGenCurves(UsdGenRegistrar &r)
{
    r.RegisterOp(TfToken("UsdGenScatter"), &CreateScatterOp);
    r.RegisterOp(TfToken("UsdGenGrow"), &CreateGrowOp);
    r.RegisterOp(TfToken("UsdGenNoise"), &CreateNoiseOp);
    r.RegisterOp(TfToken("UsdGenLength"), &CreateLengthOp);
    r.RegisterOp(TfToken("UsdGenWidth"), &CreateWidthOp);
    r.RegisterOp(TfToken("UsdGenWidthBlend"), &CreateWidthBlendOp);
    r.RegisterOp(TfToken("UsdGenCurveSource"), &CreateCurveSourceOp);
    r.RegisterOp(TfToken("UsdGenDeform"), &CreateDeformOp);
    r.RegisterOp(TfToken("UsdGenReferenceSource"), &CreateReferenceSourceOp);
    // CPU reference lane only; CUDA admission refuses these by type
    // (the capability matrix has no rows for them).
    r.RegisterOp(TfToken("UsdGenClump"), &CreateClumpOp);
    r.RegisterOp(TfToken("UsdGenGuideInterpolate"), &CreateGuideInterpolateOp);
    r.RegisterOp(TfToken("UsdGenCurl"), &CreateCurlOp);
    r.RegisterOp(TfToken("UsdGenBend"), &CreateBendOp);
    r.RegisterOp(TfToken("UsdGenWave"), &CreateWaveOp);
    r.RegisterOp(TfToken("UsdGenPart"), &CreatePartOp);
    r.RegisterOp(TfToken("UsdGenDirection"), &CreateDirectionOp);
    r.RegisterOp(TfToken("UsdGenSmooth"), &CreateSmoothOp);
    r.RegisterOp(TfToken("UsdGenResample"), &CreateResampleOp);
    r.RegisterOp(TfToken("UsdGenScale"), &CreateScaleOp);
    r.RegisterOp(TfToken("UsdGenStraighten"), &CreateStraightenOp);
    r.RegisterOp(TfToken("UsdGenDisplace"), &CreateDisplaceOp);
    r.RegisterOp(TfToken("UsdGenSculptLayer"), &CreateSculptLayerOp);
    r.RegisterOp(TfToken("UsdGenWind"), &CreateWindOp);
    r.RegisterOp(TfToken("UsdGenExprOp"), &CreateExprOp);
    r.RegisterOp(TfToken("UsdGenFreeze"), &CreateFreezeOp);
    r.RegisterOp(TfToken("UsdGenCollide"), &CreateCollideOp);
    // Whole-strand stylers and the card generator (CPU reference lane).
    r.RegisterOp(TfToken("UsdGenBraid"), &CreateBraidOp);
    r.RegisterOp(TfToken("UsdGenTwist"), &CreateTwistOp);
    r.RegisterOp(TfToken("UsdGenTrim"), &CreateTrimOp);
    r.RegisterOp(TfToken("UsdGenFrizz"), &CreateFrizzOp);
    r.RegisterOp(TfToken("UsdGenTieClump"), &CreateTieClumpOp);
    r.RegisterOp(TfToken("UsdGenCardSource"), &CreateCardSourceOp);
}
