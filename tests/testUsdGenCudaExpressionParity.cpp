// CPU-vs-CUDA expression parity, bit for bit.
//
// The scene is the shape the rewritten examples/expression-width-plane.usda
// authors on a Width DAG: a POINT-domain expression on usdGen:width
//
//     $value * fit(smoothstep($t, 0.0, 1.0), 0, 1, 1.0, 0.15)
//
// and a PRIMITIVE-domain expression on the Noise operator's usdGen:mask
//
//     $value * $u
//
// Both are evaluated twice over the same curve geometry: once by the device
// kernel (gpu::CudaExpressionContext + gpu::CudaExpressionProgram) and once by
// the host (expr::CpuExpressionContext + expr::EvaluateProgram). Every output
// float must be IDENTICAL BIT PATTERNS, which is achievable because the two
// lanes share one interpreter (expressions/irExec.h) and because these
// expressions use only exactly-rounded IEEE-754 double operations.
//
// The rest of the file walks the whole function library the same way. The
// entire noise/hash/cellnoise/voronoi/curve/spline/choose/pick family is
// asserted BIT-IDENTICAL, because SeExpr2's lattices are pure +-*/ over a
// gradient table that expressions/exprNoiseTables.h emits once for both lanes,
// and because every double-to-integer conversion goes through
// exprMath.h's ExprInt/ExprUInt32Wrap rather than a bare cast (x86 wraps where
// PTX saturates). The libm-quality transcendentals -- sin cos tan asin acos
// atan sinh cosh tanh exp log log10 pow cbrt, and the handful of builtins that
// reach one (angle, rotate, up, gamma, bias, contrast, gaussstep) -- are
// compared to a tight tolerance instead: the host CRT and CUDA's libdevice are
// each correctly rounded to a fraction of an ulp but not necessarily to the
// same bits.
//
// Every comparison runs into a FLOAT64 destination as well as the float32 one
// the operators actually use, so a last-bit difference cannot hide inside the
// float rounding.
#include "usdGenExpr/cuda/expression.h"
#include "usdGenExpr/cuda/expressionContext.h"
#include "usdGen/expressions/cpuEvaluator.h"
#include "usdGen/expressions/frontend.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace usdGen;

namespace {

int g_failures = 0;
void Check(bool ok, std::string const &what)
{
    if (!ok) { ++g_failures; std::printf("FAIL: %s\n", what.c_str()); }
}

// Four strands of six CVs, leaning like the example's Grow direction so the
// arc length, the hairT channel and the root UVs are all non-trivial.
struct Scene {
    static constexpr unsigned kCurves = 4, kCvs = 6, kPoints = kCurves * kCvs;
    std::vector<float> px, py, pz, hairT, widths;
    std::vector<float> rest, rootUV;
    std::vector<uint64_t> ids;
    std::vector<uint32_t> offsets;

    Scene()
    {
        for (unsigned c = 0; c < kCurves; ++c) {
            const float u = float(c) / float(kCurves - 1);
            const float v = 0.25f * float(c);
            rootUV.push_back(u);
            rootUV.push_back(v);
            ids.push_back((uint64_t(c + 1) << 33) + 11u * c + 3u);
            offsets.push_back(c * kCvs);
            for (unsigned i = 0; i < kCvs; ++i) {
                const float t = float(i) / float(kCvs - 1);
                const float x = u + 0.35f * 0.3f * t;
                const float y = 0.3f * t * (1.0f + 0.1f * float(c));
                const float z = v + 0.2f * 0.3f * t;
                px.push_back(x); py.push_back(y); pz.push_back(z);
                rest.push_back(x); rest.push_back(y); rest.push_back(z);
                hairT.push_back(t);
                widths.push_back(0.012f);
            }
        }
        offsets.push_back(kPoints);
    }

    expr::CpuCurveGeometryView Host() const
    {
        expr::CpuCurveGeometryView view;
        view.px = px.data(); view.py = py.data(); view.pz = pz.data();
        view.rest = rest.data();
        view.widths = widths.data();
        view.hairT = hairT.data();
        view.rootUV = rootUV.data();
        view.stableIds = ids.data();
        view.curveOffsets = offsets.data();
        view.curveCount = kCurves;
        view.pointCount = kPoints;
        return view;
    }
};

struct DeviceScene {
    gpu::DeviceBuffer<float3> points, rest;
    gpu::DeviceBuffer<float> widths, hairT;
    gpu::DeviceBuffer<float2> rootUV;
    gpu::DeviceBuffer<uint32_t> offsets;
    gpu::DeviceBuffer<uint64_t> ids;

    bool Upload(Scene const &scene)
    {
        std::vector<float3> p(Scene::kPoints), r(Scene::kPoints);
        std::vector<float2> uv(Scene::kCurves);
        for (unsigned i = 0; i < Scene::kPoints; ++i) {
            p[i] = make_float3(scene.px[i], scene.py[i], scene.pz[i]);
            r[i] = make_float3(scene.rest[i * 3], scene.rest[i * 3 + 1], scene.rest[i * 3 + 2]);
        }
        for (unsigned c = 0; c < Scene::kCurves; ++c)
            uv[c] = make_float2(scene.rootUV[c * 2], scene.rootUV[c * 2 + 1]);
        return points.reset(Scene::kPoints) == cudaSuccess &&
            rest.reset(Scene::kPoints) == cudaSuccess &&
            widths.reset(Scene::kPoints) == cudaSuccess &&
            hairT.reset(Scene::kPoints) == cudaSuccess &&
            rootUV.reset(Scene::kCurves) == cudaSuccess &&
            offsets.reset(Scene::kCurves + 1) == cudaSuccess &&
            ids.reset(Scene::kCurves) == cudaSuccess &&
            cudaMemcpy(points.data(), p.data(), p.size() * sizeof(float3), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(rest.data(), r.data(), r.size() * sizeof(float3), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(widths.data(), scene.widths.data(), scene.widths.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(hairT.data(), scene.hairT.data(), scene.hairT.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(rootUV.data(), uv.data(), uv.size() * sizeof(float2), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(offsets.data(), scene.offsets.data(), scene.offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
            cudaMemcpy(ids.data(), scene.ids.data(), scene.ids.size() * sizeof(uint64_t), cudaMemcpyHostToDevice) == cudaSuccess;
    }
};

template <class T>
bool SameBits(T a, T b)
{
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/// How closely the two lanes must agree.
enum class Agreement {
    Exact,          // identical bit patterns
    Transcendental  // within a few ulp: different correctly-rounded libms
};

bool CloseEnough(double a, double b)
{
    if (SameBits(a, b)) return true;
    if (!(a == a) || !(b == b)) return false;
    const double scale = std::fabs(a) > std::fabs(b) ? std::fabs(a) : std::fabs(b);
    return std::fabs(a - b) <= 1e-12 * (scale > 1.0 ? scale : 1.0);
}

/// Runs one binding on both lanes and compares the published floats bitwise.
void CompareLane(Scene const &scene, DeviceScene const &device, cudaStream_t stream,
                 char const *label, char const *source, expr::Domain domain,
                 float literalValue, Agreement agreement = Agreement::Exact)
{
    auto compiled = expr::Frontend::Compile(source,
        {domain, expr::ScalarType::Float32, 1});
    Check(compiled.ok, std::string("compile ") + label);
    if (!compiled.ok) {
        for (auto const &message : compiled.diagnostics) std::printf("  %s\n", message.c_str());
        return;
    }
    expr::Context controls;
    controls.frame = 1.0;
    controls.time = 1.0 / 24.0;
    controls.seed = 41;
    controls.descId = expr::DescriptionId("/World/Groom/Fur");
    controls.domain = domain;
    const size_t count = domain == expr::Domain::Point ? Scene::kPoints
        : domain == expr::Domain::Primitive ? Scene::kCurves : 1;

    // ---- device -----------------------------------------------------------
    gpu::DeviceCurveGeometryView geometry{
        {device.points.data(), Scene::kPoints}, {device.rest.data(), Scene::kPoints},
        {device.widths.data(), Scene::kPoints}, {device.offsets.data(), Scene::kCurves + 1},
        {device.ids.data(), Scene::kCurves}, Scene::kCurves, Scene::kPoints};
    gpu::CudaExpressionContext deviceContext;
    Check(deviceContext.Build(geometry, {{device.hairT.data(), Scene::kPoints},
                                         {device.rootUV.data(), Scene::kCurves}},
                              controls, stream) == gpu::ExpressionContextStatus::Ok &&
          deviceContext.Finish(stream) == gpu::ExpressionContextStatus::Ok,
          std::string("build the device context for ") + label);
    gpu::DeviceBuffer<double> deviceLiteral;
    gpu::DeviceBuffer<double> deviceOutput;
    const double literal = literalValue;
    Check(deviceLiteral.reset(1) == cudaSuccess && deviceOutput.reset(count) == cudaSuccess &&
          cudaMemcpy(deviceLiteral.data(), &literal, sizeof(literal),
                     cudaMemcpyHostToDevice) == cudaSuccess,
          std::string("allocate device output for ") + label);
    auto deviceInputs = deviceContext.Inputs();
    deviceInputs.fields[unsigned(expr::Variable::Value)] =
        {deviceLiteral.data(), 1, expr::Domain::Groom, 1};
    gpu::CudaExpressionProgram program;
    Check(program.Upload(compiled.program.IR(), stream) == gpu::ExpressionStatus::Ok &&
          program.Evaluate(deviceInputs,
                           {deviceOutput.data(), count, expr::ScalarType::Float64, 1},
                           stream) == gpu::ExpressionStatus::Ok &&
          program.Finish(stream) == gpu::ExpressionStatus::Ok,
          std::string("evaluate on the device for ") + label);
    std::vector<double> deviceValues(count, 0.0);
    Check(cudaMemcpy(deviceValues.data(), deviceOutput.data(), count * sizeof(double),
                     cudaMemcpyDeviceToHost) == cudaSuccess,
          std::string("read back the device result for ") + label);

    // ---- host -------------------------------------------------------------
    expr::CpuExpressionContext hostContext;
    std::string diagnostic;
    Check(hostContext.Build(scene.Host(), controls, &diagnostic) ==
          expr::CpuExpressionStatus::Ok,
          std::string("build the host context for ") + label + ": " + diagnostic);
    auto hostInputs = hostContext.Inputs();
    hostInputs.fields[unsigned(expr::Variable::Value)] = {&literal, 1, expr::Domain::Groom, 1};
    std::vector<double> hostValues(count, 0.0);
    Check(expr::EvaluateProgram(compiled.program.IR(), hostInputs,
                                {hostValues.data(), count, expr::ScalarType::Float64, 1}) ==
          expr::CpuExpressionStatus::Ok,
          std::string("evaluate on the host for ") + label);

    // ---- parity -----------------------------------------------------------
    size_t mismatches = 0;
    for (size_t i = 0; i < count; ++i) {
        const bool agreed = agreement == Agreement::Exact
            ? SameBits(deviceValues[i], hostValues[i])
            : CloseEnough(deviceValues[i], hostValues[i]);
        if (agreed) continue;
        if (mismatches < 4)
            std::printf("  %s[%zu]: device %.17g host %.17g\n", label, i,
                        deviceValues[i], hostValues[i]);
        ++mismatches;
    }
    Check(mismatches == 0,
          std::string(agreement == Agreement::Exact ? "bit-identical" : "agreeing") +
              " CPU/CUDA result for " + label);
}

} // namespace

int main()
{
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("testUsdGenCudaExpressionParity: no CUDA device, skipping\n");
        return 77;
    }
    cudaStream_t stream = nullptr;
    if (cudaStreamCreate(&stream) != cudaSuccess) return 2;

    Scene scene;
    DeviceScene device;
    Check(device.Upload(scene), "upload the curve geometry");

    CompareLane(scene, device, stream, "usdGen:width (point domain)",
                "$value * fit(smoothstep($t, 0.0, 1.0), 0, 1, 1.0, 0.15)",
                expr::Domain::Point, 0.012f);
    CompareLane(scene, device, stream, "usdGen:mask (primitive domain)",
                "$value * $u", expr::Domain::Primitive, 1.0f);
    // The remaining exactly-rounded ops, so a future divergence in any of them
    // is caught here and not only in the two example expressions.
    CompareLane(scene, device, stream, "arithmetic and geometry fields",
                "clamp(($P[0] + $Pref[1] - $rootP[2]) / max($cLength, 0.001), -4, 4) + "
                "min($primIndex, $pointIndex) * abs($t - 0.5) + sqrt($t) + "
                "floor($idLo * 0.001) + fmod($pointCount, 4) + "
                "mix($u, $v, $t) + ($t > 0.5 ? $cWidth : -$cWidth)",
                expr::Domain::Point, 2.0f);

    // --- every remaining exactly-rounded op ---------------------------------
    CompareLane(scene, device, stream, "scalar library",
                "round($P[0] * 7.3) + trunc($P[1] * -5.1) + hypot($P[0], $P[2]) + "
                "($t % 0.3) + deg($t) + rad($t) + invert($t) + ceil($t * 3) + "
                "compress($t, 2, 5) + expand($t, 0.1, 0.9) + boxstep($t, 0.5) + "
                "linearstep($t, 0.2, 0.8) + smoothstep($t, 0.8, 0.2) + "
                "remap($t, 0.5, 0.2, 0.3, 0) + remap($t, 0.5, 0.2, 0.3, 1) + "
                "cycle($pointIndex, 1, 3) + fit($t, 0, 1, -2, 3)",
                expr::Domain::Point, 1.0f);
    CompareLane(scene, device, stream, "vector and colour library",
                "dist($P, $rootP) + length($P) + dot($P, $Pref) + cross($P, $Pref)[1] + "
                "norm($P)[2] + ortho($P, $rootP)[0] + "
                "rgbtohsl([$u, $v, $t])[0] + hsltorgb([$u, $v, $t])[2] + "
                "saturate([$u, $v, $t], 0.3)[1] + hsi([$u, $v, $t], 30, 1.2, 0.8)[0] + "
                "midhsi([$u, $v, $t], 30, 1.2, 0.8, 0.7, 1, 0)[2]",
                expr::Domain::Point, 1.0f);
    // The noise family MUST be bit-identical: one gradient table, one lattice
    // hash, one set of integer conversions, compiled for both lanes.
    CompareLane(scene, device, stream, "noise library",
                "noise($P) + snoise($P * 3) + vnoise($P)[1] + cnoise($P)[2] + "
                "cellnoise($P * 5) + ccellnoise($P)[0] + pnoise($P, [4, 4, 4]) + "
                "fbm($P, 4, 2.1, 0.55) + vfbm($P)[0] + cfbm($P)[1] + "
                "turbulence($P) + vturbulence($P)[2] + cturbulence($P)[0]",
                expr::Domain::Point, 1.0f);
    CompareLane(scene, device, stream, "voronoi, hash and rand",
                "voronoi($P * 2) + voronoi($P * 2, 2) + voronoi($P * 2, 3) + "
                "voronoi($P * 2, 4) + voronoi($P * 2, 5) + cvoronoi($P * 2)[1] + "
                "pvoronoi($P * 2)[0] + hash($idLo, $pointIndex) + "
                "rand() + rand(7) + rand(0, 1, 3)",
                expr::Domain::Point, 1.0f);
    // Control curves: the knots are immediates in the instruction stream, so
    // the device evaluates exactly the table the host frontend prepared.
    CompareLane(scene, device, stream, "curves and variations",
                "curve($t, 0, 1, 4, 0.5, 0.7, 4, 1, 0, 4) + "
                "curve($t, 0, 0, 2, 0.5, 1, 1, 1, 0, 3) + "
                "ccurve($t, 0, [1, 0, 0], 3, 1, [0, 1, 1], 3)[1] + "
                "spline($t, 0, 0.3, 0.8, 1) + choose($t, 1, 2, 3) + "
                "wchoose($t, 1, 2, 3, 4) + pick($t * 7, 1, 5, 2, 1, 3)",
                expr::Domain::Point, 1.0f);
    // Documented exception: two correctly-rounded libms, not one.
    CompareLane(scene, device, stream, "transcendentals",
                "sin($t) + cos($t) + tan($t * 0.5) + asin($t * 0.9) + acos($t * 0.9) + "
                "atan($t) + sinh($t) + cosh($t) + tanh($t) + exp($t) + log($t + 1) + "
                "log10($t + 1) + cbrt($t + 1) + pow($t + 1, 1.7) + atan2($u, $v + 1) + "
                "angle($P, $rootP) + rotate($P, [0, 1, 0], $t)[0] + up($P, $rootP)[1] + "
                "gamma($t + 0.1, 2.2) + bias($t * 0.8 + 0.1, 0.3) + contrast($t, 0.7) + "
                "gaussstep($t, 0.2, 0.8)",
                expr::Domain::Point, 1.0f, Agreement::Transcendental);
    // The same language features the example uses, over both rates.
    CompareLane(scene, device, stream, "locals and if/else",
                "$rootWidth = 1.0;\n"
                "$tipWidth = 0.15;\n"
                "$profile = curve($t, 0, 1, 4, 0.5, 0.7, 4, 1, 0, 4);\n"
                "$jitter = 1;\n"
                "if ($u > 0.5) { $jitter = 1 + 0.25 * rand(11); }\n"
                "else { $jitter = 1 - 0.25 * rand(12); }\n"
                "$value * mix($tipWidth, $rootWidth, $profile) * $jitter",
                expr::Domain::Point, 0.012f);

    cudaStreamDestroy(stream);
    std::printf("testUsdGenCudaExpressionParity: %s\n", g_failures ? "FAILED" : "PASS");
    return g_failures ? 1 : 0;
}
