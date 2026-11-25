// usdGen — the CPU reference lane's expression evaluator.
//
// This is the host twin of gpu/expression.cu + gpu/expressionContext.cu:
//   * the interpreter itself is NOT duplicated here — both lanes call
//     expressions/irExec.h, so the arithmetic, the NaN/inf poison rules and
//     the destination range checks have exactly one definition;
//   * CpuExpressionContext builds the same per-variable fields as the device
//     kernel BuildFields, field for field, including the hairT-or-arc-length
//     $t fallback, the 64-bit id split, $u/$v from the root UVs, $cWidth,
//     $cLength, and the rule that the groom domain carries no per-element
//     field at all.
// Deliberately free of pxr types so the shape of the data (planar SoA floats
// from UsdGenCurveBuffer, or anything else) stays the caller's problem.
#ifndef USDGEN_EXPRESSIONS_CPU_EVALUATOR_H
#define USDGEN_EXPRESSIONS_CPU_EVALUATOR_H

#include "usdGen/expressions/context.h"
#include "usdGen/expressions/ir.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace usdGen::expr {

/// Member names match gpu::DeviceView so irExec.h's ReadVariable template
/// compiles unchanged against both.
struct CpuPointOwnerView {
    const uint32_t *data = nullptr;
    size_t size = 0;
};

/// Host twin of gpu::ExpressionField.
struct CpuExpressionField {
    const double *data = nullptr;
    size_t count = 0;
    Domain domain = Domain::Groom;
    unsigned components = 1;
};

class CpuExpressionSamplers;

/// Host twin of gpu::ExpressionInputs.
struct CpuExpressionInputs {
    Context context;
    size_t count = 1;
    size_t primitiveCount = 0;
    CpuPointOwnerView pointToPrimitive;
    CpuExpressionField fields[static_cast<unsigned>(Variable::CountVariables)]{};
    // Resolved IROp::Sample slots for the program being run; null when it has
    // none. Host only: the device inputs have no such member, which is what
    // keeps the device interpreter from ever sampling (irExec.h).
    CpuExpressionSamplers const *samplers = nullptr;
};

/// One IROp::Sample evaluation (expressions/samplers.cpp). `index` is the
/// element being evaluated in `inputs`' domain.
double SampleSlot(CpuExpressionSamplers const &samplers, CpuExpressionInputs const &inputs,
                  unsigned slot, double const *args, unsigned argCount, size_t index,
                  unsigned component);

/// Host twin of gpu::ExpressionOutput.
struct CpuExpressionOutput {
    void *data = nullptr;
    size_t count = 0;
    ScalarType type = ScalarType::Invalid;
    unsigned components = 1;
};

enum class CpuExpressionStatus { Ok, InvalidArgument, InvalidProgram, InvalidValue,
                                 InvalidGeometry, InvalidChannel };

/// Curve geometry in the layout the CPU engine already owns: planar positions,
/// interleaved rest/rootUV, one stable id per curve, and curveCount+1 offsets.
/// Optional channels are null when the source did not supply them.
struct CpuCurveGeometryView {
    const float *px = nullptr, *py = nullptr, *pz = nullptr;  // pointCount each
    const float *rest = nullptr;        // 3 * pointCount, interleaved xyz
    const float *widths = nullptr;      // pointCount
    const float *hairT = nullptr;       // pointCount, in [0, 1]
    const float *rootUV = nullptr;      // 2 * curveCount, interleaved uv
    // Rest root frame and surface face index, one per curve. usdGen keeps ONE
    // rest root frame, so the animated and the rest spellings of a variable
    // read the same plane: $N/$Nref <- rootN, $dPdu/$dPduref <- rootT,
    // $dPdv/$dPdvref <- rootB, $faceId <- rootPrim. A null channel allocates
    // no field at all, exactly as a null rootUV leaves $u/$v unavailable.
    const float *rootN = nullptr;       // 3 * curveCount, interleaved xyz
    const float *rootT = nullptr;       // 3 * curveCount, interleaved xyz
    const float *rootB = nullptr;       // 3 * curveCount, interleaved xyz
    const int32_t *rootPrim = nullptr;  // curveCount
    const uint64_t *stableIds = nullptr;  // curveCount
    const uint32_t *curveOffsets = nullptr;  // curveCount + 1
    size_t curveCount = 0;
    size_t pointCount = 0;
};

/// Owns the per-variable double fields for ONE evaluation domain. Rebuild it
/// whenever the geometry or the frame controls change; Inputs() borrows this
/// object's storage and must not outlive it.
class CpuExpressionContext {
public:
    CpuExpressionStatus Build(CpuCurveGeometryView const &geometry, Context context,
                              std::string *diagnostic = nullptr);
    CpuExpressionInputs const &Inputs() const noexcept { return inputs_; }
    bool Usable() const noexcept { return usable_; }

private:
    std::vector<double> fields_[static_cast<unsigned>(Variable::CountVariables)];
    std::vector<uint32_t> owners_;
    std::vector<double> arcLength_;
    CpuExpressionInputs inputs_{};
    bool usable_ = false;
};

/// Runs `program` over every element of `inputs` into `output`. Admission is
/// irExec.h's ValidProgram plus the same input/output agreement the device
/// enforces, so a program the GPU refuses is refused here too.
CpuExpressionStatus EvaluateProgram(IRProgram const &program,
                                    CpuExpressionInputs const &inputs,
                                    CpuExpressionOutput const &output);

} // namespace usdGen::expr
#endif
