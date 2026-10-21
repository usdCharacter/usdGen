// Copyright (c) 2026 Nick Burkard
// SPDX-License-Identifier: MIT
#include "usdGenImaging/rbfAnimation.h"
#include "pxr/imaging/hd/extComputationContext.h"
#include "pxr/imaging/hd/extComputationCpuCallback.h"
#include "pxr/imaging/hd/extComputationInputComputationSchema.h"
#include "pxr/imaging/hd/extComputationOutputSchema.h"
#include "pxr/imaging/hd/extComputationPrimvarSchema.h"
#include "pxr/imaging/hd/extComputationPrimvarsSchema.h"
#include "pxr/imaging/hd/extComputationSchema.h"
#include "pxr/imaging/hd/extentSchema.h"
#include "pxr/imaging/hd/primvarSchema.h"
#include "pxr/imaging/hd/primvarsSchema.h"
#include "pxr/imaging/hd/retainedDataSource.h"
#include "pxr/imaging/hd/tokens.h"
#include "pxr/imaging/hd/xformSchema.h"
#include <algorithm>
#include <cmath>
#include <limits>

PXR_NAMESPACE_USING_DIRECTIVE
namespace usdGenImaging {
namespace {
const TfToken base("ugBase"), samples("ugSamples"), coefficients("ugCoefficients"),
    centre("ugCentre"), scale("ugScale"), count("ugCount"), floatSafe("ugFloatSafe"),
    inputsName("usdGenAnimationInputs"), computeName("usdGenAnimation");
template<class T> HdDataSourceBaseHandle Value(T const& v) {
    return HdRetainedTypedSampledDataSource<T>::New(v);
}
HdContainerDataSourceHandle Container(TfToken const& name, HdDataSourceBaseHandle const& v) {
    return HdRetainedContainerDataSource::New(1, &name, &v);
}
VtVec3dArray Coefficients(usdGen::rbf::CubicField const& field) {
    size_t const n = field.SampleCount() + 4;
    VtVec3dArray result(n);
    auto const& c = field.Coefficients();
    for (size_t i = 0; i != n; ++i) result[i] = GfVec3d(c[i], c[n+i], c[2*n+i]);
    return result;
}
// Storm's compute accessor generator handles scalar doubles but currently
// broadcasts x for dvec3 inputs. Flat scalar storage preserves xyz and fp64.
template<class V> VtDoubleArray Flatten(V const& values) {
    VtDoubleArray result(values.size()*3);
    for (size_t i=0; i!=values.size(); ++i)
        for (int k=0; k!=3; ++k) result[3*i+k] = values[i][k];
    return result;
}
GfVec3d Vec(VtDoubleArray const& values, size_t i) {
    return GfVec3d(values[3*i],values[3*i+1],values[3*i+2]);
}
// Used by explicit Hydra CPU evaluation (inspection/non-GPU consumers).
// Storm executes the GLSL kernel below, without reading back CV positions.
class Callback final : public HdExtComputationCpuCallback {
    void Compute(HdExtComputationContext* ctx) override {
        auto const& b = ctx->GetInputValue(base);
        auto const& s = ctx->GetInputValue(samples);
        auto const& c = ctx->GetInputValue(coefficients);
        auto const& o = ctx->GetInputValue(centre);
        auto const& k = ctx->GetInputValue(scale);
        if (!b.IsHolding<VtVec3fArray>() || !s.IsHolding<VtDoubleArray>() ||
            !c.IsHolding<VtDoubleArray>() || !o.IsHolding<VtDoubleArray>() || !k.IsHolding<double>()) {
            ctx->RaiseComputationError(); return;
        }
        auto const& points = b.UncheckedGet<VtVec3fArray>();
        auto const& rest = s.UncheckedGet<VtDoubleArray>();
        auto const& coeff = c.UncheckedGet<VtDoubleArray>();
        size_t const n = rest.size()/3;
        if (coeff.size() != 3*(n+4) || o.UncheckedGet<VtDoubleArray>().size() != 3) {
            ctx->RaiseComputationError(); return;
        }
        VtVec3fArray out(points.size());
        double const sc = k.UncheckedGet<double>();
        for (size_t i = 0; i != points.size(); ++i) {
            GfVec3d const p(points[i]), y = (p-Vec(o.UncheckedGet<VtDoubleArray>(),0)) / sc;
            GfVec3d d = Vec(coeff,n)+Vec(coeff,n+1)*y[0]+Vec(coeff,n+2)*y[1]+Vec(coeff,n+3)*y[2];
            for (size_t j = 0; j != n; ++j) {
                double const r = (y-Vec(rest,j)).GetLength();
                d += Vec(coeff,j)*(r*r*r);
            }
            out[i] = GfVec3f(p + sc*d);
        }
        ctx->SetOutputValue(HdTokens->points, VtValue(out));
    }
};
const char* kernel = R"GLSL(
dvec3 ugCoeff(int i) {
    return dvec3(HdGet_ugCoefficients(3*i),HdGet_ugCoefficients(3*i+1),HdGet_ugCoefficients(3*i+2));
}
void compute(int index) {
    dvec3 p = dvec3(HdGet_ugBase(index));
    double s = HdGet_ugScale();
    dvec3 y = (p - dvec3(HdGet_ugCentre(0),HdGet_ugCentre(1),HdGet_ugCentre(2))) / s;
    int n = HdGet_ugCount();
    // Most rig poses are well-conditioned affine/nearly-affine fields.
    // Use fp32 for those; a conservative magnitude test selects fp64 for
    // cancellation-prone fields. Normalize in fp64 even on the fast path.
    if (HdGet_ugFloatSafe() != 0) {
        vec3 q = vec3(y);
        vec3 d = vec3(ugCoeff(n)) + vec3(ugCoeff(n+1))*q.x
            + vec3(ugCoeff(n+2))*q.y + vec3(ugCoeff(n+3))*q.z;
        vec3 correction = vec3(0);
        for (int i=0; i<n; ++i) {
            vec3 delta = q-vec3(HdGet_ugSamples(3*i),HdGet_ugSamples(3*i+1),HdGet_ugSamples(3*i+2));
            float r2 = dot(delta,delta);
            vec3 term = vec3(ugCoeff(i))*(r2*sqrt(r2))-correction;
            vec3 next = d+term;
            correction = (next-d)-term;
            d = next;
        }
        HdSet_points(index, vec3(p+s*dvec3(d)));
        return;
    }
    dvec3 d = ugCoeff(n) + ugCoeff(n+1)*y.x + ugCoeff(n+2)*y.y + ugCoeff(n+3)*y.z;
    for (int i=0; i<n; ++i) {
        dvec3 delta = y-dvec3(HdGet_ugSamples(3*i),HdGet_ugSamples(3*i+1),HdGet_ugSamples(3*i+2));
        double r2 = dot(delta,delta);
        // One double Newton correction of a float square-root seed avoids
        // the expensive fp64 sqrt on consumer GPUs while recovering about
        // 46 significant bits. Extreme distances retain the native sqrt.
        double r;
        if (r2 > 1e-30 && r2 < 1e30) {
            r = double(sqrt(float(r2)));
            r += fma(-r,r,r2)*double(0.5*inversesqrt(float(r2)));
        } else {
            r = sqrt(r2);
        }
        d += ugCoeff(i)*(r2*r);
    }
    HdSet_points(index, vec3(p+s*d));
}
)GLSL";

HdContainerDataSourceHandle Compute(SdfPath const& path, SdfPath const& coeffPath, size_t size, int sampleCount) {
    const TfToken names[] = {base, samples, centre, scale, coefficients, floatSafe};
    HdDataSourceBaseHandle links[6];
    for (size_t i = 0; i != 6; ++i)
        links[i] = HdExtComputationInputComputationSchema::Builder()
            .SetSourceComputation(HdRetainedTypedSampledDataSource<SdfPath>::New(
                i >= 4 ? coeffPath : path.AppendChild(inputsName)))
            .SetSourceComputationOutputName(HdRetainedTypedSampledDataSource<TfToken>::New(names[i])).Build();
    auto output = HdExtComputationOutputSchema::Builder().SetValueType(
        HdRetainedTypedSampledDataSource<HdTupleType>::New({HdTypeFloatVec3, 1})).Build();
    HdDataSourceBaseHandle outputs[] = {output};
    TfToken outputName = HdTokens->points;
    return Container(HdExtComputationSchema::GetSchemaToken(),
        HdExtComputationSchema::Builder()
            // Storm includes the computation's own input range in execution;
            // keep its scalar dispatch metadata here so that range is valid.
            .SetInputValues(Container(count, Value(sampleCount)))
            .SetInputComputations(HdExtComputationInputComputationContainerSchema::BuildRetained(6,names,links))
            .SetOutputs(HdExtComputationOutputContainerSchema::BuildRetained(1,&outputName,outputs))
            .SetGlslKernel(HdRetainedTypedSampledDataSource<std::string>::New(kernel))
            .SetCpuCallback(HdRetainedTypedSampledDataSource<HdExtComputationCpuCallbackSharedPtr>::New(
                std::make_shared<Callback>()))
            .SetDispatchCount(HdRetainedTypedSampledDataSource<size_t>::New(size))
            .SetElementCount(HdRetainedTypedSampledDataSource<size_t>::New(size)).Build());
}

// Taylor bound for F(box). Summing its Jacobian before taking absolute
// values retains cancellation in an RBF fit. Independent radial intervals
// can inflate a centimetre-sized tile to metres under a head rotation.
// The cubic radial Hessian has norm <= 6*r, bounding the remainder below.
// Work is proportional to sparse drivers, never to the number of hair CVs.
GfRange3d Bounds(GfRange3d const& box, usdGen::rbf::CubicField const& field) {
    GfVec3d const middle = (box.GetMin()+box.GetMax())*0.5;
    GfVec3d const half = box.GetSize()*0.5;
    GfVec3d const y = (middle-field.Centre())/field.Scale();
    double const radius = half.GetLength()/field.Scale();
    auto const c = Coefficients(field);
    size_t const n = field.SampleCount();
    GfVec3d jacobian[] = {c[n+1]+GfVec3d(1,0,0), c[n+2]+GfVec3d(0,1,0), c[n+3]+GfVec3d(0,0,1)};
    GfVec3d remainder(0);
    for (size_t i=0; i!=n; ++i) {
        GfVec3d const delta = y-field.Samples()[i];
        double const r = delta.GetLength();
        for (int k=0; k!=3; ++k) {
            jacobian[k] += c[i]*(3*r*delta[k]);
            remainder[k] += std::abs(c[i][k])*(r+radius);
        }
    }
    GfVec3d const center = middle+field.Displacement(middle);
    GfVec3d extent = remainder*(3*field.Scale()*radius*radius);
    for (int k=0; k!=3; ++k) {
        for (int j=0; j!=3; ++j) extent[k] += std::abs(jacobian[j][k])*half[j];
        extent[k] += 1e-5*std::max(1.0,box.GetSize().GetLength()) +
            4*std::numeric_limits<float>::epsilon()*std::abs(center[k]);
    }
    return GfRange3d(center-extent, center+extent);
}
bool FloatSafe(GfRange3d const& box, usdGen::rbf::CubicField const& field) {
    // Bound absolute term magnitudes over the entire hair domain, including
    // a unit guard for sample/query rounding near zero distance. Large
    // coefficients or extrapolation keep the full-precision kernel.
    GfVec3d const a = (box.GetMin()-field.Centre())/field.Scale();
    GfVec3d const b = (box.GetMax()-field.Centre())/field.Scale();
    auto const c = Coefficients(field);
    size_t const n = field.SampleCount();
    GfVec3d magnitude;
    for (int k=0; k!=3; ++k) {
        magnitude[k] = std::abs(c[n][k]);
        for (int j=0; j!=3; ++j)
            magnitude[k] += std::abs(c[n+1+j][k])*(1+std::max(std::abs(a[j]),std::abs(b[j])));
    }
    for (size_t i=0; i!=n; ++i) {
        GfVec3d far;
        for (int k=0; k!=3; ++k)
            far[k] = std::max(std::abs(a[k]-field.Samples()[i][k]),std::abs(b[k]-field.Samples()[i][k]));
        double const radius = far.GetLength()+1;
        for (int k=0; k!=3; ++k) magnitude[k] += std::abs(c[i][k])*radius*radius*radius;
    }
    double const roundoff = (2*n+32)*std::numeric_limits<float>::epsilon()*field.Scale();
    double const tolerance = 1e-5*std::max(1.0,box.GetSize().GetLength());
    return std::isfinite(magnitude.GetLength()) && magnitude.GetLength()*roundoff <= tolerance;
}
bool Pose(RbfAnimation* result, usdGen::rbf::CubicField const& field, GfMatrix4d const& xform) {
    GfRange3d domain;
    for (auto const& bounds : result->restBounds) domain.UnionWith(bounds.second);
    TfToken const inputNames[] = {coefficients,floatSafe};
    HdDataSourceBaseHandle const inputs[] = {Value(Flatten(Coefficients(field))),Value(int(FloatSafe(domain,field)))};
    result->computations[result->coefficientsPath] = Container(HdExtComputationSchema::GetSchemaToken(),
        HdExtComputationSchema::Builder().SetInputValues(HdRetainedContainerDataSource::New(2,inputNames,inputs)).Build());
    for (auto const& tile : result->binding->tiles) {
        auto const extent = Bounds(result->restBounds.at(tile.path), field);
        for (int k=0; k!=3; ++k)
            if (!std::isfinite(extent.GetMin()[k]) || !std::isfinite(extent.GetMax()[k]) ||
                std::abs(extent.GetMin()[k]) > std::numeric_limits<float>::max() ||
                std::abs(extent.GetMax()[k]) > std::numeric_limits<float>::max()) return false;
        auto primvar = HdExtComputationPrimvarSchema::Builder()
            .SetInterpolation(HdExtComputationPrimvarSchema::BuildInterpolationDataSource(HdPrimvarSchemaTokens->vertex))
            .SetRole(HdExtComputationPrimvarSchema::BuildRoleDataSource(HdPrimvarSchemaTokens->point))
            .SetSourceComputation(HdRetainedTypedSampledDataSource<SdfPath>::New(tile.path.AppendChild(computeName)))
            .SetSourceComputationOutputName(HdRetainedTypedSampledDataSource<TfToken>::New(HdTokens->points))
            .SetValueType(HdRetainedTypedSampledDataSource<HdTupleType>::New({HdTypeFloatVec3,1})).Build();
        TfToken const names[] = {HdPrimvarsSchema::GetSchemaToken(), HdExtComputationPrimvarsSchema::GetSchemaToken(),
            HdExtentSchema::GetSchemaToken(), HdXformSchema::GetSchemaToken()};
        HdDataSourceBaseHandle const values[] = {
            Container(HdTokens->points, HdBlockDataSource::New()), Container(HdTokens->points, primvar),
            HdExtentSchema::Builder().SetMin(HdRetainedTypedSampledDataSource<GfVec3d>::New(extent.GetMin()))
                .SetMax(HdRetainedTypedSampledDataSource<GfVec3d>::New(extent.GetMax())).Build(),
            HdXformSchema::Builder().SetMatrix(HdRetainedTypedSampledDataSource<GfMatrix4d>::New(xform)).Build()};
        result->overlays[tile.path] = HdRetainedContainerDataSource::New(4,names,values);
    }
    return true;
}
}
std::shared_ptr<const RbfAnimation> BuildRbfAnimation(
    std::shared_ptr<const usdGen::UsdGenRbfAnimationBinding> const& binding, GfMatrix4d const& xform) {
    if (!binding || binding->tiles.empty()) return {};
    auto result = std::make_shared<RbfAnimation>();
    result->binding = binding;
    result->coefficientsPath = binding->tiles.front().path.GetParentPath().AppendChild(TfToken("usdGenAnimationCoefficients"));
    auto const& field = binding->field;
    auto const rest = Flatten(field.Samples());
    auto const origin = field.Centre();
    for (auto const& tile : binding->tiles) {
        GfRange3d bounds;
        for (auto const& p : tile.restPoints) bounds.UnionWith(GfVec3d(p));
        // Conservative width padding; the exact authored padding is retained
        // in the binding and is independent of pose.
        bounds.SetMin(bounds.GetMin()-GfVec3d(tile.widthPadding));
        bounds.SetMax(bounds.GetMax()+GfVec3d(tile.widthPadding));
        result->restBounds.emplace(tile.path, bounds);
        TfToken const names[] = {base,samples,centre,scale};
        HdDataSourceBaseHandle const values[] = {Value(tile.restPoints),Value(rest),Value(VtDoubleArray{origin[0],origin[1],origin[2]}),
            Value(field.Scale())};
        result->computations[tile.path.AppendChild(inputsName)] = Container(HdExtComputationSchema::GetSchemaToken(),
            HdExtComputationSchema::Builder().SetInputValues(HdRetainedContainerDataSource::New(4,names,values)).Build());
        result->computations[tile.path.AppendChild(computeName)] = Compute(tile.path,result->coefficientsPath,tile.restPoints.size(),int(field.SampleCount()));
    }
    if (!Pose(result.get(),field,xform)) return {};
    return result;
}
std::shared_ptr<const RbfAnimation> PoseRbfAnimation(RbfAnimation const& previous, usdGen::UsdGenGraphDesc const& desc) {
    auto const& binding = *previous.binding;
    auto surface = std::find_if(desc.surfaces.begin(),desc.surfaces.end(),[&](auto const& s) {return s.path==binding.surface;});
    if (surface == desc.surfaces.end()) return {};
    auto field = binding.field;
    GfMatrix4d const relative = surface->worldMatrix * desc.xformMatrix.GetInverse();
    std::vector<GfVec3d> now;
    now.reserve(binding.sampleIndices.size());
    for (size_t i : binding.sampleIndices) {
        if (i >= surface->points.size()) return {};
        now.push_back(relative.Transform(GfVec3d(surface->points[i])));
    }
    std::string error;
    if (!field.Solve(now,&error)) return {};
    auto result = std::make_shared<RbfAnimation>(previous);
    if (!Pose(result.get(),field,desc.xformMatrix)) return {};
    return result;
}
}
