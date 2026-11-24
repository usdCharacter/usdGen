#include "usdGen/expressions/cpuEvaluator.h"

#include "usdGen/expressions/irExec.h"

#include <cmath>
#include <limits>

namespace usdGen::expr {
namespace {

bool Finite3(const float *p, size_t i)
{
    return ExprFinite(p[i * 3 + 0]) && ExprFinite(p[i * 3 + 1]) && ExprFinite(p[i * 3 + 2]);
}

bool Fail(std::string *diagnostic, char const *message)
{
    if (diagnostic) *diagnostic = message;
    return false;
}

/// Mirrors the device Validate kernel in gpu/expressionContext.cu.
bool ValidateGeometry(CpuCurveGeometryView const &g, std::string *diagnostic,
                      bool *badChannel)
{
    *badChannel = false;
    if (!g.curveOffsets || g.curveOffsets[0] != 0 ||
        g.curveOffsets[g.curveCount] != g.pointCount)
        return Fail(diagnostic, "expression geometry curve offsets are malformed");
    for (size_t c = 0; c < g.curveCount; ++c) {
        const uint32_t a = g.curveOffsets[c], b = g.curveOffsets[c + 1];
        if (a >= b || b > g.pointCount)
            return Fail(diagnostic, "expression geometry has an empty or out-of-range curve");
        if (!g.stableIds)
            return Fail(diagnostic, "expression geometry has no stable curve ids");
    }
    for (size_t i = 0; i < g.pointCount; ++i) {
        if (!g.px || !g.py || !g.pz ||
            !ExprFinite(g.px[i]) || !ExprFinite(g.py[i]) || !ExprFinite(g.pz[i]))
            return Fail(diagnostic, "expression geometry has a non-finite point");
        if (g.rest && !Finite3(g.rest, i))
            return Fail(diagnostic, "expression geometry has a non-finite rest point");
        if (g.widths && !ExprFinite(g.widths[i]))
            return Fail(diagnostic, "expression geometry has a non-finite width");
        if (g.hairT && (!ExprFinite(g.hairT[i]) || g.hairT[i] < 0.0f || g.hairT[i] > 1.0f)) {
            *badChannel = true;
            return Fail(diagnostic, "expression hairT channel is outside [0, 1]");
        }
    }
    if (g.hairT) for (size_t c = 0; c < g.curveCount; ++c) {
        const uint32_t begin = g.curveOffsets[c], end = g.curveOffsets[c + 1];
        if (begin < end && end <= g.pointCount && end > begin + 1 &&
            (g.hairT[begin] != 0.0f || g.hairT[end - 1] != 1.0f)) {
            *badChannel = true;
            return Fail(diagnostic, "expression hairT channel does not run 0 -> 1 per curve");
        }
    }
    if (g.rootUV) for (size_t c = 0; c < g.curveCount; ++c)
        if (!ExprFinite(g.rootUV[c * 2 + 0]) || !ExprFinite(g.rootUV[c * 2 + 1])) {
            *badChannel = true;
            return Fail(diagnostic, "expression rootUV channel is non-finite");
        }
    // The rest root frame planes. rootPrim is an integer face index and is
    // deliberately not checked for finiteness, exactly as on the device.
    if (g.rootN) for (size_t c = 0; c < g.curveCount; ++c)
        if (!Finite3(g.rootN, c)) {
            *badChannel = true;
            return Fail(diagnostic, "expression rootN channel is non-finite");
        }
    if (g.rootT) for (size_t c = 0; c < g.curveCount; ++c)
        if (!Finite3(g.rootT, c)) {
            *badChannel = true;
            return Fail(diagnostic, "expression rootT channel is non-finite");
        }
    if (g.rootB) for (size_t c = 0; c < g.curveCount; ++c)
        if (!Finite3(g.rootB, c)) {
            *badChannel = true;
            return Fail(diagnostic, "expression rootB channel is non-finite");
        }
    return true;
}

} // namespace

CpuExpressionStatus CpuExpressionContext::Build(CpuCurveGeometryView const &g,
                                                Context context,
                                                std::string *diagnostic)
{
    usable_ = false;
    inputs_ = CpuExpressionInputs{};
    if (!ExprIsDomain(context.domain)) {
        if (diagnostic) *diagnostic = "invalid expression evaluation domain";
        return CpuExpressionStatus::InvalidArgument;
    }
    if (!std::isfinite(context.frame) || !std::isfinite(context.time)) {
        if (diagnostic) *diagnostic = "expression frame/time controls are non-finite";
        return CpuExpressionStatus::InvalidArgument;
    }
    inputs_.context = context;
    inputs_.count = 1;
    // The groom domain evaluates once and carries no per-element field: every
    // spatial variable is unavailable there (Registry::Validate rejects them),
    // so there is nothing to build. This mirrors BuildImpl's early return.
    if (context.domain == Domain::Groom) {
        inputs_.context.count = 1;
        usable_ = true;
        return CpuExpressionStatus::Ok;
    }
    if ((g.curveCount == 0) != (g.pointCount == 0)) {
        if (diagnostic) *diagnostic = "expression geometry has curves without points";
        return CpuExpressionStatus::InvalidGeometry;
    }
    bool badChannel = false;
    if (g.curveCount && !ValidateGeometry(g, diagnostic, &badChannel))
        return badChannel ? CpuExpressionStatus::InvalidChannel
                          : CpuExpressionStatus::InvalidGeometry;

    const size_t n = context.domain == Domain::Primitive ? g.curveCount : g.pointCount;
    inputs_.count = n;
    inputs_.primitiveCount = g.curveCount;
    inputs_.context.count = static_cast<uint32_t>(n);

    // Owner map and cumulative arc length: the host twin of BuildOwnersAndArc.
    owners_.assign(g.pointCount, 0);
    arcLength_.assign(g.pointCount, 0.0);
    for (size_t c = 0; c < g.curveCount; ++c) {
        const uint32_t begin = g.curveOffsets[c], end = g.curveOffsets[c + 1];
        double total = 0.0;
        for (uint32_t p = begin; p < end; ++p) {
            owners_[p] = static_cast<uint32_t>(c);
            if (p > begin) {
                const double dx = double(g.px[p]) - g.px[p - 1];
                const double dy = double(g.py[p]) - g.py[p - 1];
                const double dz = double(g.pz[p]) - g.pz[p - 1];
                total += std::sqrt(dx * dx + dy * dy + dz * dz);
            }
            arcLength_[p] = total;
        }
    }
    inputs_.pointToPrimitive = {owners_.empty() ? nullptr : owners_.data(), g.pointCount};

    const bool point = context.domain == Domain::Point;
    auto allocate = [&](Variable v, unsigned components) -> double * {
        const unsigned index = static_cast<unsigned>(v);
        fields_[index].assign(n * components, 0.0);
        inputs_.fields[index] = {fields_[index].empty() ? nullptr : fields_[index].data(),
                                 n, context.domain, components};
        return fields_[index].empty() ? nullptr : fields_[index].data();
    };
    // Exactly the field set gpu::CudaExpressionContext::BuildImpl materializes:
    // P/RootP always, PRef/RootPRef only with a rest channel, CWidth only with
    // widths, U/V only with root UVs, N/NRef, DPdu/DPduRef, DPdv/DPdvRef and
    // FaceId only with their root-frame channel, PointIndex/PointCount only at
    // point rate.
    double *P = allocate(Variable::P, 3);
    double *rootP = allocate(Variable::RootP, 3);
    double *pRef = g.rest ? allocate(Variable::PRef, 3) : nullptr;
    double *rootPRef = g.rest ? allocate(Variable::RootPRef, 3) : nullptr;
    double *cWidth = g.widths ? allocate(Variable::CWidth, 1) : nullptr;
    double *t = allocate(Variable::T, 1);
    double *primIndex = allocate(Variable::PrimIndex, 1);
    double *primCount = allocate(Variable::PrimCount, 1);
    double *cLength = allocate(Variable::CLength, 1);
    double *idLo = allocate(Variable::IdLo, 1);
    double *idHi = allocate(Variable::IdHi, 1);
    double *id = allocate(Variable::Id, 1);
    double *u = g.rootUV ? allocate(Variable::U, 1) : nullptr;
    double *v = g.rootUV ? allocate(Variable::V, 1) : nullptr;
    double *faceId = g.rootPrim ? allocate(Variable::FaceId, 1) : nullptr;
    double *N = g.rootN ? allocate(Variable::N, 3) : nullptr;
    double *NRef = g.rootN ? allocate(Variable::NRef, 3) : nullptr;
    double *dPdu = g.rootT ? allocate(Variable::DPdu, 3) : nullptr;
    double *dPduRef = g.rootT ? allocate(Variable::DPduRef, 3) : nullptr;
    double *dPdv = g.rootB ? allocate(Variable::DPdv, 3) : nullptr;
    double *dPdvRef = g.rootB ? allocate(Variable::DPdvRef, 3) : nullptr;
    double *pointIndex = point ? allocate(Variable::PointIndex, 1) : nullptr;
    double *pointCount = point ? allocate(Variable::PointCount, 1) : nullptr;

    auto set3 = [](double *dst, size_t i, float x, float y, float z) {
        if (!dst) return;
        dst[i * 3 + 0] = x; dst[i * 3 + 1] = y; dst[i * 3 + 2] = z;
    };
    for (size_t i = 0; i < n; ++i) {
        const size_t c = point ? owners_[i] : i;
        const uint32_t root = g.curveOffsets[c];
        const size_t sample = point ? i : size_t(root);
        set3(P, i, g.px[sample], g.py[sample], g.pz[sample]);
        set3(rootP, i, g.px[root], g.py[root], g.pz[root]);
        if (g.rest) {
            set3(pRef, i, g.rest[sample * 3], g.rest[sample * 3 + 1], g.rest[sample * 3 + 2]);
            set3(rootPRef, i, g.rest[root * 3], g.rest[root * 3 + 1], g.rest[root * 3 + 2]);
        }
        if (cWidth) cWidth[i] = g.widths[sample];
        // $t: the authored hairT channel when present, otherwise the curve's
        // normalized cumulative arc length — the device T() helper verbatim.
        if (!point) {
            t[i] = 0.0;
        } else if (g.hairT) {
            t[i] = g.hairT[i];
        } else {
            const double total = arcLength_[g.curveOffsets[c + 1] - 1];
            t[i] = total > 0.0 ? arcLength_[i] / total : 0.0;
        }
        primIndex[i] = double(c);
        primCount[i] = double(g.curveCount);
        cLength[i] = arcLength_[g.curveOffsets[c + 1] - 1];
        const uint64_t stable = g.stableIds[c];
        idLo[i] = double(uint32_t(stable));
        idHi[i] = double(uint32_t(stable >> 32));
        id[i] = double(stable);
        if (u) { u[i] = g.rootUV[c * 2 + 0]; v[i] = g.rootUV[c * 2 + 1]; }
        // Per-curve root data, addressed by the owning curve at either rate.
        if (faceId) faceId[i] = double(g.rootPrim[c]);
        if (N) {
            set3(N, i, g.rootN[c * 3], g.rootN[c * 3 + 1], g.rootN[c * 3 + 2]);
            set3(NRef, i, g.rootN[c * 3], g.rootN[c * 3 + 1], g.rootN[c * 3 + 2]);
        }
        if (dPdu) {
            set3(dPdu, i, g.rootT[c * 3], g.rootT[c * 3 + 1], g.rootT[c * 3 + 2]);
            set3(dPduRef, i, g.rootT[c * 3], g.rootT[c * 3 + 1], g.rootT[c * 3 + 2]);
        }
        if (dPdv) {
            set3(dPdv, i, g.rootB[c * 3], g.rootB[c * 3 + 1], g.rootB[c * 3 + 2]);
            set3(dPdvRef, i, g.rootB[c * 3], g.rootB[c * 3 + 1], g.rootB[c * 3 + 2]);
        }
        if (pointIndex) {
            pointIndex[i] = double(i - root);
            pointCount[i] = double(g.curveOffsets[c + 1] - root);
        }
    }
    usable_ = true;
    return CpuExpressionStatus::Ok;
}

CpuExpressionStatus EvaluateProgram(IRProgram const &program,
                                    CpuExpressionInputs const &inputs,
                                    CpuExpressionOutput const &output)
{
    if (!ValidProgram(program)) return CpuExpressionStatus::InvalidProgram;
    const unsigned outputCount = program.outputCount ? program.outputCount : 1u;
    // The same agreement gpu::ValidInputs enforces before a launch.
    if (!ExprIsDomain(inputs.context.domain) || inputs.count != output.count ||
        (inputs.count && !output.data) || output.type == ScalarType::Invalid ||
        output.components != outputCount ||
        (inputs.context.domain == Domain::Groom && inputs.count != 1))
        return CpuExpressionStatus::InvalidArgument;
    for (auto const &field : inputs.fields) {
        if (!field.data && field.count == 0) continue;
        if (!field.data || !ExprIsDomain(field.domain) || field.components == 0 ||
            field.components > 4)
            return CpuExpressionStatus::InvalidArgument;
        size_t expected = field.domain == Domain::Groom ? 1 : inputs.count;
        if (field.domain == Domain::Primitive && inputs.context.domain == Domain::Point) {
            expected = inputs.primitiveCount;
            if (!inputs.pointToPrimitive.data || inputs.pointToPrimitive.size != inputs.count)
                return CpuExpressionStatus::InvalidArgument;
        } else if (field.domain != Domain::Groom && field.domain != inputs.context.domain)
            return CpuExpressionStatus::InvalidArgument;
        if (field.count != expected) return CpuExpressionStatus::InvalidArgument;
    }
    uint16_t results[4]{};
    for (unsigned c = 0; c < outputCount; ++c)
        results[c] = program.outputCount ? program.output[c] : program.result;

    double registers[kExprRegisters];
    bool valid = true;
    for (size_t i = 0; i < inputs.count; ++i) {
        ExecuteElement(program.instructions.data(), program.instructions.size(),
                       registers, inputs, i);
        for (unsigned c = 0; c < outputCount; ++c)
            if (!Store(registers[results[c]], output, i * outputCount + c)) valid = false;
    }
    return valid ? CpuExpressionStatus::Ok : CpuExpressionStatus::InvalidValue;
}

} // namespace usdGen::expr
